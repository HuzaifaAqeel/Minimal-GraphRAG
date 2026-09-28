"""Graph retrieval: link a question to entities, expand, collect triples."""

from __future__ import annotations

import re

import networkx as nx

from .extract import Triple, _entities_in_sentence  # noqa: F401  (shared heuristic)


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def link_entities(graph: nx.DiGraph, question: str) -> list[str]:
    """Find graph nodes mentioned in the question (exact + token overlap)."""
    mentions = _entities_in_sentence(question)
    nodes = list(graph.nodes())
    lowered = {n: _normalize(n) for n in nodes}
    linked: list[str] = []

    for mention in mentions:
        m = _normalize(mention)
        # 1) exact match
        for node, ln in lowered.items():
            if ln == m and node not in linked:
                linked.append(node)
        # 2) substring either way ("Aurora" -> "Aurora Robotics")
        for node, ln in lowered.items():
            if node in linked:
                continue
            if (m in ln or ln in m) and len(m) > 3:
                linked.append(node)
    # 3) single content-word overlap fallback
    if not linked:
        q_tokens = {t for t in re.findall(r"[a-z]{4,}", question.lower())}
        scored = []
        for node, ln in lowered.items():
            overlap = len(q_tokens & set(re.findall(r"[a-z]{4,}", ln)))
            if overlap:
                scored.append((overlap, node))
        scored.sort(reverse=True)
        linked = [n for _, n in scored[:3]]
    return linked


def expand_subgraph(
    graph: nx.DiGraph, seeds: list[str], hops: int = 2, max_nodes: int = 40
) -> nx.DiGraph:
    """Breadth-first expansion around seed entities (undirected walk)."""
    seen: set[str] = set(seeds)
    frontier = list(seeds)
    for _ in range(hops):
        nxt: list[str] = []
        for node in frontier:
            if node not in graph:
                continue
            for nbr in list(graph.successors(node)) + list(graph.predecessors(node)):
                if nbr not in seen and len(seen) < max_nodes:
                    seen.add(nbr)
                    nxt.append(nbr)
        frontier = nxt
        if not frontier:
            break
    return graph.subgraph(seen).copy()


def triples_from_subgraph(subgraph: nx.DiGraph, seeds: list[str]) -> list[Triple]:
    """Collect triples, seed-adjacent edges first."""
    seed_set = set(seeds)
    scored: list[tuple[int, Triple]] = []
    for u, v, data in subgraph.edges(data=True):
        for relation in data.get("relations", ["related_to"]):
            score = 0
            if u in seed_set:
                score += 2
            if v in seed_set:
                score += 2
            scored.append((score, Triple(u, relation, v)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [t for _, t in scored]


def retrieve(
    graph: nx.DiGraph, question: str, hops: int = 2
) -> tuple[list[str], list[Triple], nx.DiGraph]:
    """Full retrieval step: link -> expand -> collect."""
    seeds = link_entities(graph, question)
    if not seeds:
        return [], [], nx.DiGraph()
    subgraph = expand_subgraph(graph, seeds, hops=hops)
    return seeds, triples_from_subgraph(subgraph, seeds), subgraph
