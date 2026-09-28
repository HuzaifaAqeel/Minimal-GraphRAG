"""Knowledge-graph storage on top of NetworkX."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from .extract import Triple


def build_graph(triples: list[Triple]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for triple in triples:
        graph.add_node(triple.subject, kind="entity")
        graph.add_node(triple.object, kind="entity")
        if graph.has_edge(triple.subject, triple.object):
            # merge duplicate relations instead of overwriting
            existing = graph[triple.subject][triple.object].get("relations", [])
            if triple.relation not in existing:
                existing.append(triple.relation)
        else:
            graph.add_edge(triple.subject, triple.object, relations=[triple.relation])
    return graph


def save_graph(graph: nx.DiGraph, path: str | Path) -> Path:
    path = Path(path)
    payload = {
        "nodes": sorted(graph.nodes()),
        "edges": [
            {"source": u, "target": v, "relations": d.get("relations", [])}
            for u, v, d in graph.edges(data=True)
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_graph(path: str | Path) -> nx.DiGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = nx.DiGraph()
    graph.add_nodes_from(payload["nodes"], kind="entity")
    for edge in payload["edges"]:
        graph.add_edge(edge["source"], edge["target"], relations=edge["relations"])
    return graph


def graph_stats(graph: nx.DiGraph) -> dict:
    degrees = dict(graph.degree())
    top = sorted(degrees.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "top_entities": [{"entity": name, "degree": deg} for name, deg in top],
    }
