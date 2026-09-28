"""Streamlit app: visualize the knowledge graph and query it interactively."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from graphrag.answer import answer
from graphrag.graph import load_graph
from graphrag.retrieve import retrieve

st.set_page_config(page_title="Minimal GraphRAG", layout="wide")
st.title("Minimal GraphRAG — knowledge-graph explorer")

graph_path = st.sidebar.text_input("Graph file", value="graph.json")
hops = st.sidebar.slider("Expansion hops", 1, 3, 2)
dry_run = st.sidebar.checkbox("Dry-run (no LLM)", value=True)

if not Path(graph_path).exists():
    st.warning(
        f"Graph file `{graph_path}` not found. Build one first:\n\n"
        "`python -m graphrag.cli build --docs data --dry-run`"
    )
    st.stop()

graph: nx.DiGraph = load_graph(graph_path)
question = st.text_input(
    "Ask a question",
    value="Who founded Aurora Robotics?",
    key="question",
)

col_graph, col_answer = st.columns([3, 2])

with col_graph:
    st.subheader("Graph")
    highlight: set[str] = set()
    if question:
        seeds, triples, subgraph = retrieve(graph, question, hops=hops)
    else:
        seeds, triples, subgraph = [], [], graph
    view = subgraph if question and len(subgraph) > 0 else graph
    highlight = set(seeds)

    fig, ax = plt.subplots(figsize=(9, 6))
    pos = nx.spring_layout(view, seed=42, k=0.9)
    colors = ["#ff7f0e" if n in highlight else "#1f77b4" for n in view.nodes()]
    nx.draw_networkx_nodes(view, pos, node_color=colors, node_size=700, ax=ax)
    nx.draw_networkx_edges(view, pos, edge_color="#bbbbbb", arrows=True, ax=ax,
                           arrowsize=12, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(view, pos, font_size=8, ax=ax)
    edge_labels = {
        (u, v): ", ".join(d.get("relations", [])) for u, v, d in view.edges(data=True)
    }
    nx.draw_networkx_edge_labels(view, pos, edge_labels=edge_labels, font_size=6, ax=ax)
    ax.set_axis_off()
    st.pyplot(fig, use_container_width=True)
    st.caption(f"Showing {view.number_of_nodes()} nodes / {view.number_of_edges()} edges")

with col_answer:
    st.subheader("Answer")
    if question:
        if not seeds:
            st.info("No entities from the graph matched the question.")
        else:
            st.markdown(f"**Linked entities:** {', '.join(seeds)}")
            st.markdown(answer(question, triples, dry_run=dry_run))
            with st.expander("Retrieved triples"):
                for t in triples[:25]:
                    st.code(f"{t.subject} --[{t.relation}]--> {t.object}")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
)
