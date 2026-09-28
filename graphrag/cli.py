"""CLI: build a knowledge graph from documents, query it, inspect it."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from graphrag.answer import answer
from graphrag.extract import extract_triples
from graphrag.graph import build_graph, graph_stats, load_graph, save_graph
from graphrag.retrieve import retrieve

load_dotenv()

DEFAULT_GRAPH = "graph.json"
TEXT_EXTENSIONS = {".txt", ".md", ".rst"}


def _iter_documents(docs_dir: Path):
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            yield path


def cmd_build(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs)
    if not docs_dir.is_dir():
        print(f"error: docs directory not found: {docs_dir}", file=sys.stderr)
        return 1
    all_triples = []
    files = list(_iter_documents(docs_dir))
    if not files:
        print(f"error: no .txt/.md files under {docs_dir}", file=sys.stderr)
        return 1
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        triples = extract_triples(text, dry_run=args.dry_run)
        all_triples.extend(triples)
        print(f"  {path.name}: {len(triples)} triples")
    graph = build_graph(all_triples)
    save_graph(graph, args.output)
    stats = graph_stats(graph)
    print(f"\nBuilt graph -> {args.output}")
    print(f"  nodes: {stats['nodes']}, edges: {stats['edges']}")
    print("  top entities: " + ", ".join(
        f"{e['entity']} ({e['degree']})" for e in stats["top_entities"]
    ))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    graph = load_graph(args.graph)
    seeds, triples, _ = retrieve(graph, args.question, hops=args.hops)
    print(f"Linked entities: {', '.join(seeds) if seeds else '(none)'}")
    print(f"Retrieved {len(triples)} triples\n")
    print(answer(args.question, triples, dry_run=args.dry_run))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    graph = load_graph(args.graph)
    print(json.dumps(graph_stats(graph), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphrag",
        description="Minimal GraphRAG: knowledge-graph Q&A over your documents.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the LLM; use deterministic heuristics (works offline).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --dry-run is accepted both before and after the subcommand
    def add_dry_run(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="Skip the LLM; use deterministic heuristics (works offline).",
        )

    build_p = sub.add_parser("build", help="Extract triples and build the graph.")
    build_p.add_argument("--docs", default="data", help="Directory of .txt/.md files.")
    build_p.add_argument("--output", default=DEFAULT_GRAPH, help="Where to save the graph.")
    add_dry_run(build_p)
    build_p.set_defaults(func=cmd_build)

    query_p = sub.add_parser("query", help="Ask a question over the graph.")
    query_p.add_argument("question", help="Natural-language question.")
    query_p.add_argument("--graph", default=DEFAULT_GRAPH)
    query_p.add_argument("--hops", type=int, default=2)
    add_dry_run(query_p)
    query_p.set_defaults(func=cmd_query)

    stats_p = sub.add_parser("stats", help="Show graph statistics.")
    stats_p.add_argument("--graph", default=DEFAULT_GRAPH)
    add_dry_run(stats_p)
    stats_p.set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
