"""
state.py
--------
The shared data structure every LangGraph node reads from and writes to.

Nodes never call each other directly - they all communicate through this one
TypedDict. Each node returns a partial dict of the fields it changed, and
LangGraph merges that into the running state (using each field's reducer,
where one is declared - see `messages` below).
"""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Full running conversation. `add_messages` is a LangGraph reducer: new
    # messages are APPENDED to this list, never used to overwrite it.
    messages: Annotated[list, add_messages]

    # The user's most recent raw request, kept for convenience/logging.
    user_request: str

    # The tool_calls the LLM most recently requested (plan_node's output).
    plan: list

    # Every ToolMessage produced so far this turn.
    tool_results: list

    # Facts pulled from long-term memory relevant to this turn.
    memories: list

    # Free-text scratch field a node can use to describe what it's doing.
    current_task: str

    # Bumped once per PLAN -> ACT -> OBSERVE cycle. should_continue uses this
    # together with MAX_ITERATIONS to guarantee the loop always terminates.
    iteration_count: int

    # Set once the graph is ready to answer. main.py prints this.
    final_response: str

    # Human-in-the-loop bookkeeping (see graph.py's RISKY_TOOLS handling).
    pending_confirmation: Optional[dict]
    confirmed: bool