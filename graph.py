"""
graph.py
--------
The LangGraph agent loop. This is the heart of JARVIS: everything else
(agent.py's LLM, tools.py's tools, memory.py's storage) is wired together
here into UNDERSTAND -> PLAN -> ACT -> OBSERVE -> (loop or RESPOND).

    understand --> plan --(conditional: should_continue)--> act --> observe --\
                    ^                                                          |
                    |                                                          |
                    +---------------------(loop back)-------------------------+
                    |
                    +--(no more tool calls needed)--> respond --> END

PLAN is where the LLM actually reasons: every time it runs, it sees the
whole conversation (including the results of any tools already run this
turn) and decides whether it needs to call more tools or is ready to
answer. That single decision point is what makes re-planning "free" - it's
the same node running again with fresher context, not a separate feature.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from agent import llm, llm_with_tools
from config import MAX_ITERATIONS
from prompts import SYSTEM_PROMPT
from state import AgentState
from tools import RISKY_TOOLS, TOOLS_BY_NAME, recall_memory


# ------------------------------------------------------------------- nodes --
def understand_node(state: AgentState) -> dict:
    """Figure out what the user is asking, and pull any relevant long-term
    memories into context before we start planning."""
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_request = human_msgs[-1].content if human_msgs else ""

    memories = recall_memory.invoke({"query": user_request}) if user_request else []

    return {
        "user_request": user_request,
        "memories": memories,
        "current_task": "understanding the request",
    }


def plan_node(state: AgentState) -> dict:
    """Ask the LLM what to do next. It sees the full conversation so far
    (including any tool results from earlier iterations this turn) and
    either requests more tool calls or produces a final answer."""
    system_content = SYSTEM_PROMPT
    if state.get("memories"):
        facts = "\n".join(f"- {m}" for m in state["memories"])
        system_content += f"\n\nRelevant remembered facts about the user:\n{facts}"

    messages_for_llm = [SystemMessage(content=system_content)] + state["messages"]
    response = llm_with_tools.invoke(messages_for_llm)

    return {
        "messages": [response],
        "plan": response.tool_calls or [],
        "iteration_count": state.get("iteration_count", 0) + 1,
        "current_task": "planning",
    }


def act_node(state: AgentState) -> dict:
    """Execute every tool call the plan requested. Risky tools are
    intercepted and require explicit confirmation before running; any tool
    that raises an exception reports the error back as its result instead
    of crashing the whole agent."""
    results = []
    pending = None
    confirmed = state.get("confirmed", False)

    for call in state["plan"]:
        name, args, call_id = call["name"], call.get("args", {}), call["id"]

        if name in RISKY_TOOLS and not confirmed:
            try:
                count = len(TOOLS_BY_NAME["get_tasks"].invoke({}))
            except Exception:
                count = "an unknown number of"
            pending = call
            results.append(
                ToolMessage(
                    content=(
                        f"This action would permanently affect all {count} tasks. "
                        "NOT executed - awaiting explicit user confirmation (yes/no)."
                    ),
                    tool_call_id=call_id,
                )
            )
            continue

        try:
            output = TOOLS_BY_NAME[name].invoke(args)
            results.append(ToolMessage(content=str(output), tool_call_id=call_id))
        except Exception as e:
            results.append(
                ToolMessage(content=f"Tool '{name}' failed: {e}", tool_call_id=call_id)
            )

    return {
        "messages": results,
        "tool_results": results,
        "pending_confirmation": pending,
        # Once used, a confirmation is spent - the next risky call needs a fresh "yes".
        "confirmed": False,
        "current_task": "acting",
    }


def observe_node(state: AgentState) -> dict:
    """Read the tool results back into state. Deliberately thin: the real
    decision ('do I need to act again?') is made by the LLM the next time
    plan_node runs, not here."""
    return {"current_task": "observing"}


def respond_node(state: AgentState) -> dict:
    """Produce the final answer. If the loop stopped because the LLM gave
    a plain answer (no tool_calls), that answer IS the response - no extra
    LLM call needed. Otherwise (a pending confirmation, or the iteration
    cap was hit mid-tool-call) we make one more call to phrase things
    naturally from whatever tool results exist so far.

    That wrap-up call uses llm_with_tools, not the plain llm: we are forcing
    a stop either way, but a history full of prior tool calls can nudge the
    model into requesting yet another one. Groq's API rejects a tool call
    from a request that declared zero tools (a hard 400 error) - keeping
    tools bound makes that same response merely unused, not invalid. If the
    model responds with no text at all, we fall back to a deterministic
    summary instead of returning an empty reply.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None):
        final = last.content
    else:
        wrap_up = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        final = wrap_up.content or _fallback_summary(state)
    return {"final_response": final, "current_task": "responding"}


def _fallback_summary(state: AgentState) -> str:
    """Deterministic text used only if the wrap-up call comes back empty."""
    if state.get("pending_confirmation"):
        return "I need an explicit yes or no before I can proceed with that action."
    recent = state.get("tool_results", [])[-3:]
    if recent:
        details = "; ".join(str(r.content) for r in recent)
        return f"I've reached my step limit for this turn. Here's what I found: {details}"
    return "I've reached my step limit for this turn without a final answer - could you rephrase?"


# -------------------------------------------------------- conditional edge --
def should_continue(state: AgentState) -> str:
    """The router. Everything 'agentic' about JARVIS comes down to this
    function: keep acting, or wrap up and respond?"""
    if state.get("pending_confirmation"):
        return "respond"
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "respond"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "act"
    return "respond"


# ----------------------------------------------------------------- wiring --
workflow = StateGraph(AgentState)
workflow.add_node("understand", understand_node)
workflow.add_node("plan", plan_node)
workflow.add_node("act", act_node)
workflow.add_node("observe", observe_node)
workflow.add_node("respond", respond_node)

workflow.set_entry_point("understand")
workflow.add_edge("understand", "plan")
workflow.add_conditional_edges("plan", should_continue, {"act": "act", "respond": "respond"})
workflow.add_edge("act", "observe")
workflow.add_edge("observe", "plan")  # the loop back that makes re-planning "free"
workflow.add_edge("respond", END)

app = workflow.compile()


def new_turn_state(prior_state: dict | None, user_text: str) -> dict:
    """Build the input state for one turn. We keep the full message history
    ourselves and pass it in fresh each call - simple enough for a workshop,
    no LangGraph checkpointer required."""
    prior_messages = prior_state["messages"] if prior_state else []
    prior_memories = prior_state.get("memories", []) if prior_state else []
    return {
        "messages": prior_messages + [HumanMessage(content=user_text)],
        "user_request": user_text,
        "plan": [],
        "tool_results": [],
        "memories": prior_memories,
        "current_task": "",
        "iteration_count": 0,
        "final_response": "",
        "pending_confirmation": None,
        "confirmed": False,
    }


CONFIRM_WORDS = {"yes", "y", "yeah", "yep", "confirm", "do it"}


def is_confirmation(text: str) -> bool:
    return text.strip().lower() in CONFIRM_WORDS


def resolve_pending_confirmation(state: dict, user_text: str) -> dict:
    """Handle the user's yes/no answer to a pending human-in-the-loop
    confirmation - deterministically, without asking the LLM to re-decide.

    Why not just let the normal graph handle "yes"? Because the model only
    ever sees its OWN earlier tool request plus a ToolMessage saying "NOT
    executed". A model can (and in testing, sometimes did) reply "Done!"
    without actually re-emitting the tool call - a confident-sounding
    hallucination of success on exactly the operation we most need to get
    right. Executing the already-approved call directly removes that risk
    entirely; the LLM is only used afterwards, to phrase the result.
    """
    pending = state["pending_confirmation"]
    user_msg = HumanMessage(content=user_text)

    if not is_confirmation(user_text):
        # Not a clear "yes" - cancel the pending action, but still let the
        # graph handle whatever the user actually said; it may be a new,
        # unrelated request rather than a plain "no".
        cancelled = {**state, "pending_confirmation": None, "confirmed": False}
        return app.invoke(new_turn_state(cancelled, user_text))

    try:
        output = TOOLS_BY_NAME[pending["name"]].invoke(pending.get("args", {}))
    except Exception as e:
        output = f"Tool '{pending['name']}' failed: {e}"

    tool_msg = ToolMessage(content=str(output), tool_call_id=pending["id"])
    messages = state["messages"] + [user_msg, tool_msg]
    wrap_up = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    final = wrap_up.content or str(output)

    return {
        **state,
        "messages": messages,
        "final_response": final,
        "pending_confirmation": None,
        "confirmed": False,
        "tool_results": [tool_msg],
    }


if __name__ == "__main__":
    import sys

    try:  # see main.py for why this guard exists (Windows console encoding)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # python graph.py -> a couple of scripted turns, printed end to end.
    state = app.invoke(new_turn_state(None, "What time is it?"))
    print("JARVIS:", state["final_response"])

    state = app.invoke(new_turn_state(state, "Remember that I prefer studying in the morning."))
    print("JARVIS:", state["final_response"])