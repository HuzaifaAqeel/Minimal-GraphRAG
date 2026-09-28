"""Smoke tests for the dry-run pipeline (no API key needed)."""

from graphrag.answer import answer_dry_run
from graphrag.extract import extract_dry_run
from graphrag.graph import build_graph, graph_stats, load_graph, save_graph
from graphrag.retrieve import link_entities, retrieve

SAMPLE = (
    "Aurora Robotics was founded by Dr. Lena Marsh in 2021. "
    "Dr. Lena Marsh works at Aurora Robotics. "
    "Aurora Robotics is based in Berlin. "
    "Aurora Robotics develops Scout."
)


def test_extract_finds_founded_by():
    triples = extract_dry_run(SAMPLE)
    relations = {(t.subject, t.relation, t.object) for t in triples}
    assert ("Aurora Robotics", "founded_by", "Dr. Lena Marsh") in relations


def test_extract_finds_works_at_and_develops():
    triples = extract_dry_run(SAMPLE)
    relations = {(t.subject, t.relation, t.object) for t in triples}
    assert ("Dr. Lena Marsh", "works_at", "Aurora Robotics") in relations
    assert ("Aurora Robotics", "develops", "Scout") in relations


def test_graph_roundtrip(tmp_path):
    triples = extract_dry_run(SAMPLE)
    graph = build_graph(triples)
    assert graph.number_of_nodes() >= 4
    path = tmp_path / "g.json"
    save_graph(graph, path)
    loaded = load_graph(path)
    assert loaded.number_of_nodes() == graph.number_of_nodes()
    assert loaded.number_of_edges() == graph.number_of_edges()


def test_retrieve_links_and_expands():
    graph = build_graph(extract_dry_run(SAMPLE))
    seeds = link_entities(graph, "Who founded Aurora Robotics?")
    assert "Aurora Robotics" in seeds
    seeds, triples, subgraph = retrieve(graph, "Who founded Aurora Robotics?")
    assert any(t.relation == "founded_by" for t in triples)
    assert subgraph.number_of_nodes() > 1


def test_answer_mentions_triple():
    graph = build_graph(extract_dry_run(SAMPLE))
    _, triples, _ = retrieve(graph, "Who founded Aurora Robotics?")
    out = answer_dry_run("Who founded Aurora Robotics?", triples)
    assert "Dr. Lena Marsh" in out
    assert "founded_by" in out


def test_stats_shape():
    graph = build_graph(extract_dry_run(SAMPLE))
    stats = graph_stats(graph)
    assert stats["nodes"] == graph.number_of_nodes()
    assert stats["edges"] == graph.number_of_edges()
    assert stats["top_entities"]
