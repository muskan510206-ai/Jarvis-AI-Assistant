"""
agent.py
--------
The "brain": a Groq-hosted LLM, wrapped by LangChain, with every tool bound
to it so it can request them by name. graph.py's plan_node calls
`llm_with_tools.invoke(...)` - everything about *how* the loop runs lives
there; this file only builds the model.
"""

from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL, require_groq_key
from tools import TOOLS

require_groq_key()

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.3)

llm_with_tools = llm.bind_tools(TOOLS)


if __name__ == "__main__":
    # python agent.py -> a couple of quick, visible sanity checks.
    reply = llm.invoke("Say hello as JARVIS, in one short sentence.")
    print("Plain reply:", reply.content)

    tool_reply = llm_with_tools.invoke("What time is it?")
    print("Tool calls requested:", tool_reply.tool_calls)