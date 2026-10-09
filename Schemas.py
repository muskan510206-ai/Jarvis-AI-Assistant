"""
schemas.py
----------
Pydantic models used to force structured, validated output from the LLM.

RouterDecision isn't wired into the final graph (JARVIS's real routing comes
from tool-calling + should_continue in graph.py) - it exists here as the
teaching example from Section 4 ("Structured Output"), and it's genuinely
useful any time you want a yes/no + reasoning decision you can trust the
shape of. Run this file directly to see it in action.
"""

from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    """A structured decision about whether a request needs tools at all."""

    needs_tools: bool = Field(
        description="True if answering the request requires calling a tool."
    )
    reasoning: str = Field(
        description="One short sentence explaining the decision."
    )


if __name__ == "__main__":
    from config import require_groq_key, GROQ_MODEL, GROQ_API_KEY
    from langchain_groq import ChatGroq

    require_groq_key()
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
    structured_llm = llm.with_structured_output(RouterDecision)

    result = structured_llm.invoke("What's the weather in Pune?")
    print(result)
    assert isinstance(result, RouterDecision)