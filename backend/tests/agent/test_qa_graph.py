"""
End-to-end integration test for the QA subgraph.

Run with:
    cd backend && uv run pytest tests/agent/test_qa_graph.py -v -s

The -s flag lets print statements through so you can see each node's output.
Qdrant is mocked so no live services are required — only the LLM API key.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.core.schema import S2Paper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_PAPER = S2Paper(
    paperId="paper_attention_001",
    title="Attention Is All You Need",
    abstract=(
        "We propose a new simple network architecture, the Transformer, based solely on "
        "attention mechanisms, dispensing with recurrence and convolutions entirely. "
        "The model uses multi-head self-attention with scaled dot-product attention. "
        "Experiments on two machine translation tasks show these models to be superior "
        "in quality while being more parallelizable and requiring significantly less "
        "time to train."
    ),
    authors=[{"name": "Vaswani"}, {"name": "Shazeer"}],
    year=2017,
    citationCount=50000,
)

MOCK_EVIDENCE = [
    Document(
        page_content=(
            "The Transformer uses multi-head attention which allows the model to jointly "
            "attend to information from different representation subspaces at different positions. "
            "For each head, we compute scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V."
        ),
        metadata={"id": "paper_attention_001", "para": "3"},
    ),
    Document(
        page_content=(
            "Multi-Head Attention consists of h attention heads running in parallel. "
            "Each head performs attention on a different learned linear projection of "
            "queries, keys, and values."
        ),
        metadata={"id": "paper_attention_001", "para": "4"},
    ),
]

INITIAL_STATE = {
    "user_query": "What attention mechanism does the Transformer use and how does it work?",
    "papers": [MOCK_PAPER],
    "selected_paper_ids": ["paper_attention_001"],
    "messages": [
        HumanMessage(
            content="What attention mechanism does the Transformer use and how does it work?"
        )
    ],
    "evidences": [],
    "qa_iteration": 0,
    "sufficient_evidence": False,
    "limitation": "",
    "final_answer": "",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_qa_graph_end_to_end(capsys):
    """
    Run the full QA graph with a mock Qdrant backend.

    Uses graph.stream() to print each node's state transitions,
    giving the same visibility as a playground file but repeatably.
    """
    from app.agent.qa import qa_graph

    mock_qdrant = MagicMock()
    mock_qdrant.search_selected_ids.return_value = MOCK_EVIDENCE

    with patch("app.tools.search.QdrantService", return_value=mock_qdrant):
        steps = []
        for step in qa_graph.stream(INITIAL_STATE):
            node_name = list(step.keys())[0]
            node_output = step[node_name]
            steps.append((node_name, node_output))

            print(f"\n{'='*60}")
            print(f"NODE: {node_name}")
            if isinstance(node_output, dict):
                print(f"  evidences count : {len(node_output.get('evidences', []))}")
                print(f"  sufficient      : {node_output.get('sufficient_evidence')}")
                print(f"  qa_iteration    : {node_output.get('qa_iteration')}")
                print(f"  limitation      : {node_output.get('limitation', '')[:80]}")
                if node_output.get("final_answer"):
                    print(f"  final_answer    :\n{node_output['final_answer']}")

    node_names = [name for name, _ in steps]
    final_state = steps[-1][1]

    # Graph must have visited qa_retrieve, tools, qa_evaluate, qa_answer
    assert "qa_retrieve" in node_names, "qa_retrieve node never ran"
    assert "tools" in node_names, "tools node never ran"
    assert "qa_evaluate" in node_names, "qa_evaluate node never ran"
    assert "qa_answer" in node_names, "qa_answer node never ran"

    # Graph must produce a final answer
    assert final_state.get("final_answer"), "Expected a non-empty final_answer"
