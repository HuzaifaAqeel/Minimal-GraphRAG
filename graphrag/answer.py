"""Answer generation from retrieved triples."""

from __future__ import annotations

import os

from .extract import Triple


def _format_context(triples: list[Triple], limit: int = 25) -> str:
    lines = [f"- {t.subject} --[{t.relation}]--> {t.object}" for t in triples[:limit]]
    return "\n".join(lines)


def answer_dry_run(question: str, triples: list[Triple]) -> str:
    """Deterministic answer: report the most relevant triples found."""
    if not triples:
        return (
            "I couldn't find any entities in the knowledge graph matching "
            "your question. Try asking about one of the indexed entities."
        )
    context = _format_context(triples)
    return (
        f"Based on the knowledge graph, here is what I found for: \"{question}\"\n\n"
        f"{context}\n\n"
        f"({len(triples)} fact(s) retrieved from the graph; "
        "dry-run mode lists raw triples instead of an LLM summary.)"
    )


def answer_with_llm(
    question: str, triples: list[Triple], model: str = "gemini-2.5-flash"
) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set; use dry-run mode instead.")
    from google import genai  # lazy import: optional dependency

    client = genai.Client(api_key=api_key)
    prompt = (
        "Answer the question using ONLY the facts below. "
        "If the facts don't contain the answer, say so.\n\n"
        f"Facts:\n{_format_context(triples)}\n\nQuestion: {question}"
    )
    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def answer(question: str, triples: list[Triple], *, dry_run: bool = False) -> str:
    if dry_run or not os.environ.get("GOOGLE_API_KEY"):
        return answer_dry_run(question, triples)
    try:
        return answer_with_llm(question, triples)
    except Exception:
        return answer_dry_run(question, triples)
