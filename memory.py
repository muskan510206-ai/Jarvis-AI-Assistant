"""
memory.py
---------
JARVIS's long-term memory: facts that survive across sessions, stored on
disk in SQLite (not just in this turn's `state["messages"]`).

Keeping this in its own file means tools.py's remember_fact/recall_memory
stay thin wrappers, and the storage/search implementation can be swapped
(e.g. for ChromaDB's semantic search) without touching anything else.
"""

import datetime
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    return conn


def store_fact(fact: str) -> int:
    """Permanently remember a fact. Returns the new row's id."""
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO facts (fact, created_at) VALUES (?, ?)",
            (fact, datetime.datetime.now().isoformat()),
        )
        return cur.lastrowid


def search_facts(query: str, limit: int = 5) -> list[str]:
    """Keyword search over remembered facts.

    This is intentionally simple (good enough for a workshop, and fully
    understandable in five lines). To upgrade to real semantic search:

        import chromadb
        client = chromadb.PersistentClient(path=str(DB_PATH.parent / "chroma"))
        collection = client.get_or_create_collection("facts")
        collection.add(documents=[fact], ids=[str(new_id)])
        ...
        results = collection.query(query_texts=[query], n_results=limit)

    That's the only change required - remember_fact/recall_memory in
    tools.py don't need to know which one is running underneath.
    """
    words = [w.lower() for w in query.split() if len(w) > 2]
    with _conn() as conn:
        rows = conn.execute(
            "SELECT fact FROM facts ORDER BY id DESC"
        ).fetchall()
    all_facts = [row[0] for row in rows]
    if not words:
        return all_facts[:limit]
    matches = [f for f in all_facts if any(w in f.lower() for w in words)]
    return matches[:limit] if matches else all_facts[:limit]


def all_facts() -> list[str]:
    with _conn() as conn:
        rows = conn.execute("SELECT fact FROM facts ORDER BY id DESC").fetchall()
    return [row[0] for row in rows]


if __name__ == "__main__":
    store_fact("Prefers studying in the morning")
    print("Stored. All facts:", all_facts())
    print("Search 'study preference':", search_facts("study preference"))