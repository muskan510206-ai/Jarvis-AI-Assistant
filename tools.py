"""
tools.py
--------
Every tool JARVIS can call. Each is a plain Python function decorated with
@tool - LangChain reads the type hints + docstring to build a JSON schema
the LLM can see and request. The LLM never runs these functions itself; it
only asks for them (see graph.py's act_node), which is what makes error
handling and human-in-the-loop confirmation possible.
"""

import ast
import json
import operator
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from memory import search_facts, store_fact

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Tools in this set are consequential/hard-to-undo. act_node (graph.py)
# intercepts them and requires an explicit "yes" from the user first.
RISKY_TOOLS = {"delete_all_tasks"}


# ---------------------------------------------------------------- storage --
def _load(filename: str) -> list:
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(filename: str, data: list) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ------------------------------------------------------------- time / math --
@tool
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M")


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numbers are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12*7+3'. Supports
    + - * / ** % and parentheses. Never use this for anything but math."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


# ------------------------------------------------------------------ weather --
# Mock data - the goal of this workshop is agent architecture, not wiring up
# a real weather API. Swapping this for a real HTTP call later doesn't
# change anything else about JARVIS.
_MOCK_WEATHER = {
    "pune": {"condition": "Sunny", "temp_c": 31},
    "mumbai": {"condition": "Humid, light rain", "temp_c": 28},
    "delhi": {"condition": "Hazy", "temp_c": 34},
    "bengaluru": {"condition": "Pleasant, overcast", "temp_c": 24},
    "bangalore": {"condition": "Pleasant, overcast", "temp_c": 24},
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (mock data for this workshop)."""
    info = _MOCK_WEATHER.get(city.strip().lower())
    if not info:
        return f"Error: no weather data for '{city}'. Try Pune, Mumbai, Delhi, or Bengaluru."
    return f"{city.title()}: {info['condition']}, {info['temp_c']}C"


# -------------------------------------------------------------------- tasks --
@tool
def get_tasks() -> list:
    """Return all tasks, including completed ones."""
    return _load("tasks.json")


@tool
def create_task(title: str, due_date: str) -> str:
    """Create a new task. due_date must be in YYYY-MM-DD format."""
    tasks = _load("tasks.json")
    new_id = max((t["id"] for t in tasks), default=0) + 1
    tasks.append({"id": new_id, "title": title, "due_date": due_date, "done": False})
    _save("tasks.json", tasks)
    return f"Created task #{new_id}: {title} (due {due_date})"


@tool
def complete_task(task_id: int) -> str:
    """Mark a task as done, given its id."""
    tasks = _load("tasks.json")
    for t in tasks:
        if t["id"] == task_id:
            t["done"] = True
            _save("tasks.json", tasks)
            return f"Marked task #{task_id} ('{t['title']}') as done."
    return f"Error: no task with id {task_id}."


@tool
def delete_all_tasks() -> str:
    """Permanently delete every task. This is DESTRUCTIVE and irreversible -
    the caller must have already confirmed this with the user."""
    tasks = _load("tasks.json")
    _save("tasks.json", [])
    return f"Deleted all {len(tasks)} tasks."


# ----------------------------------------------------------------- calendar --
@tool
def get_calendar() -> list:
    """Return all calendar events."""
    return _load("calendar.json")


@tool
def create_calendar_event(title: str, date: str, time: str) -> str:
    """Add a new calendar event. date is YYYY-MM-DD, time is HH:MM (24h)."""
    events = _load("calendar.json")
    events.append({"title": title, "date": date, "time": time})
    _save("calendar.json", events)
    return f"Added '{title}' on {date} at {time}"


# ------------------------------------------------------------------- memory --
@tool
def remember_fact(fact: str) -> str:
    """Permanently remember a fact the user tells you, for use in future
    sessions (e.g. preferences, recurring details)."""
    store_fact(fact)
    return f"Got it, I'll remember: {fact}"


@tool
def recall_memory(query: str) -> list:
    """Search previously remembered facts relevant to a query."""
    return search_facts(query)


# --------------------------------------------------------------------- web --
@tool
def search_web(query: str) -> str:
    """Search the web for information not covered by other tools.
    (Mocked for this workshop - swap in a real search API to go further.)"""
    return (
        f"[mock search result for '{query}'] Web search isn't wired to a "
        "real API in this workshop - see the Final Student Challenge."
    )


TOOLS = [
    get_current_time,
    calculate,
    get_weather,
    get_tasks,
    create_task,
    complete_task,
    delete_all_tasks,
    get_calendar,
    create_calendar_event,
    remember_fact,
    recall_memory,
    search_web,
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


if __name__ == "__main__":
    print("Registered tools:", list(TOOLS_BY_NAME.keys()))
    print(get_current_time.invoke({}))
    print(calculate.invoke({"expression": "12*7+3"}))