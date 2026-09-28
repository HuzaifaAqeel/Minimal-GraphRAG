# Minimal-GraphRAG

A tiny, from-scratch implementation of **GraphRAG** (the Microsoft Research idea):
turn a pile of documents into a **knowledge graph** of entities and relations,
then answer questions by **retrieving over the graph** instead of raw text chunks.

Why a graph? Plain RAG retrieves isolated passages. A knowledge graph connects
the dots — *who founded what, who works where, what was built when* — so
multi-hop questions ("Who founded the company that built Scout?") resolve by
walking edges instead of hoping one chunk contains the whole answer.

## How it works

```
documents/ ──extract──▶ triples ──build──▶ NetworkX graph ──retrieve──▶ answer
   .txt/.md    (LLM or        (subject,        (DiGraph,        (entity linking
                heuristics)   relation,         saved as JSON)   + k-hop expansion
                              object)                            + LLM/dry-run answer)
```

1. **Extract** (`graphrag/extract.py`) — Gemini pulls `(subject, relation, object)`
   triples from each document. No API key? A deterministic heuristic extractor
   (capitalized-phrase entities, verb-pattern relations, co-occurrence fallback)
   keeps everything working offline.
2. **Build** (`graphrag/graph.py`) — triples become a directed NetworkX graph,
   persisted as a single JSON file.
3. **Retrieve** (`graphrag/retrieve.py`) — link question mentions to graph nodes,
   expand 1–3 hops, collect and rank triples by seed proximity.
4. **Answer** (`graphrag/answer.py`) — Gemini answers from the retrieved triples;
   dry-run mode returns the ranked triples directly.
5. **Visualize** (`visualize.py`) — Streamlit app: interactive graph view with the
   retrieved subgraph highlighted, side-by-side with the answer.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add GOOGLE_API_KEY for the LLM backends (optional)

# 1. Build the graph from the sample docs (dry-run works with no key)
python -m graphrag.cli build --docs data --dry-run

# 2. Ask questions
python -m graphrag.cli query "Who founded Aurora Robotics?" --dry-run
python -m graphrag.cli query "Which companies did Lena Marsh found?" --dry-run

# 3. Graph stats
python -m graphrag.cli stats

# 4. Visual explorer
streamlit run visualize.py
```

With `GOOGLE_API_KEY` set, drop `--dry-run` and extraction/answering switch to
`gemini-2.5-flash`.

## Project structure

```
├── graphrag/
│   ├── cli.py        # build / query / stats commands
│   ├── extract.py    # LLM + heuristic triple extraction
│   ├── graph.py      # NetworkX build / save / load / stats
│   ├── retrieve.py   # entity linking + k-hop expansion
│   └── answer.py     # LLM + dry-run answer generation
├── visualize.py      # Streamlit graph explorer
├── data/             # sample documents (swap in your own .txt/.md)
└── tests/            # pytest suite (runs fully offline)
```

## Notes

- The graph is a plain JSON edge list — inspectable, diffable, versionable.
- Heuristic extraction is deliberately simple (it's a *minimal* GraphRAG);
  the LLM path is where production quality comes from.
- Tested with Python 3.11+.

## License

MIT — see [LICENSE](LICENSE).
