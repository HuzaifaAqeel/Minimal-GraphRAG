"""Entity/relation extraction from raw text.

Two backends:
- ``llm``: asks Gemini to return JSON triples (needs GOOGLE_API_KEY).
- ``dry_run``: deterministic heuristics (capitalized phrases as entities,
  verb-mediated patterns as relations, sentence co-occurrence as fallback).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "it",
    "its", "this", "that", "these", "those", "as", "which", "who", "their",
    "she", "he", "they", "we", "you", "i", "him", "her", "them", "us",
}

# Shared entity fragment: capitalized words, tolerating titles like "Dr."
_ENT = r"[A-Z][\w&'\-]*(?:\.?\s+[A-Z][\w&'\-]+)*"

# Matches runs of capitalized words: "Aurora Robotics", "Dr. Lena Marsh"
_ENTITY_RE = re.compile(rf"\b({_ENT})\b")

# "X is a Y" / "X was founded by Y" style patterns
_IS_A_RE = re.compile(
    rf"({_ENT})\s+is\s+(?:a|an|the)\s+([a-z][\w\s\-]*?)(?:\.|,|;|$)",
)
_FOUNDED_BY_RE = re.compile(rf"({_ENT})\s+was\s+founded\s+by\s+({_ENT})")
_WORKS_AT_RE = re.compile(rf"({_ENT})\s+(?:works|worked)\s+at\s+({_ENT})")
_LOCATED_IN_RE = re.compile(rf"({_ENT})\s+is\s+(?:located|based)\s+in\s+({_ENT})")
_DEVELOPS_RE = re.compile(
    rf"({_ENT})\s+(?:develops|developed|builds|built|launched|released)\s+"
    rf"(?:the\s+|a\s+|an\s+)?({_ENT})"
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Abbreviations whose period must not trigger a sentence split ("Dr. Lena").
_ABBR_PERIOD = "\u0001"
_ABBREVIATIONS = (
    "Dr.", "Mr.", "Mrs.", "Ms.", "St.", "Jr.", "Sr.", "Prof.",
    "Inc.", "Ltd.", "vs.",
)


def _split_sentences(text: str) -> list[str]:
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _ABBR_PERIOD))
    return [p.replace(_ABBR_PERIOD, ".") for p in _SENT_SPLIT_RE.split(protected)]


def _clean_entity(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" .,'\"")
    return name


def _entities_in_sentence(sentence: str) -> list[str]:
    found: list[str] = []
    for match in _ENTITY_RE.finditer(sentence):
        name = _clean_entity(match.group(1))
        words = name.split()
        # drop leading stopwords / single lowercase-ish noise
        while words and words[0].lower() in _STOPWORDS:
            words.pop(0)
        if not words:
            continue
        name = " ".join(words)
        if len(name) < 2 or name.lower() in _STOPWORDS:
            continue
        # skip sentence-initial single common words ("The company ...")
        if len(words) == 1 and match.start(1) == 0 and words[0].lower() in {"the", "a", "an"}:
            continue
        if name not in found:
            found.append(name)
    return found


def extract_dry_run(text: str) -> list[Triple]:
    """Deterministic heuristic extraction -- no API key needed."""
    triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()

    def add(s: str, r: str, o: str) -> None:
        s, o = _clean_entity(s), _clean_entity(o)
        if not s or not o or s.lower() == o.lower():
            return
        key = (s, r, o)
        if key not in seen:
            seen.add(key)
            triples.append(Triple(s, r, o))

    for sentence in _split_sentences(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for pattern, relation in (
            (_FOUNDED_BY_RE, "founded_by"),
            (_WORKS_AT_RE, "works_at"),
            (_LOCATED_IN_RE, "located_in"),
            (_DEVELOPS_RE, "develops"),
        ):
            for m in pattern.finditer(sentence):
                add(m.group(1), relation, m.group(2))
        for m in _IS_A_RE.finditer(sentence):
            add(m.group(1), "is_a", m.group(2).strip())

        # fallback: every pair of entities co-occurring in a sentence
        ents = _entities_in_sentence(sentence)
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                add(ents[i], "related_to", ents[j])
    return triples


def extract_with_llm(text: str, model: str = "gemini-2.5-flash") -> list[Triple]:
    """Extract triples with Gemini. Requires GOOGLE_API_KEY."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set; use dry-run mode instead.")
    from google import genai  # lazy import: optional dependency

    client = genai.Client(api_key=api_key)
    prompt = (
        "Extract factual knowledge triples from the text below. "
        "Return ONLY a JSON array of objects with keys "
        '"subject", "relation", "object". Use short snake_case relations. '
        "Subjects and objects should be named entities or concise noun phrases.\n\n"
        f"Text:\n{text[:12000]}"
    )
    response = client.models.generate_content(model=model, contents=prompt)
    raw = (response.text or "").strip()
    # tolerate code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    items = json.loads(raw)
    triples = []
    for item in items:
        try:
            triples.append(
                Triple(
                    subject=str(item["subject"]).strip(),
                    relation=str(item["relation"]).strip().replace(" ", "_").lower(),
                    object=str(item["object"]).strip(),
                )
            )
        except (KeyError, TypeError):
            continue
    return triples


def extract_triples(text: str, *, dry_run: bool = False) -> list[Triple]:
    """Extract triples, using the LLM when a key is available."""
    if dry_run or not os.environ.get("GOOGLE_API_KEY"):
        return extract_dry_run(text)
    try:
        return extract_with_llm(text)
    except Exception:
        # fall back to heuristics rather than failing the whole build
        return extract_dry_run(text)
