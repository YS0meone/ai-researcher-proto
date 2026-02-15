"""
Integration tests for _optimize_for_search.

Verifies that keyword-style search queries are correctly derived from
multi-turn conversations containing unresolved references.

Run with:
    cd backend && uv run pytest tests/agent/test_optimize_for_search.py -v -s
    LANGSMITH_TEST_SUITE="Query Optimization - Search" uv run pytest tests/agent/test_optimize_for_search.py -v -s
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import testing as t

from app.agent.graph import _optimize_for_search


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

PRONOUN_REF_MESSAGES = [
    HumanMessage(content="Find me papers about the Transformer architecture by Vaswani et al."),
    AIMessage(content="I found the paper 'Attention Is All You Need' by Vaswani et al. (2017)."),
    HumanMessage(content="What attention mechanism does it use?"),
]

METHOD_REF_MESSAGES = [
    HumanMessage(content="I'm looking for papers on contrastive learning, specifically SimCLR."),
    AIMessage(content="I found 'A Simple Framework for Contrastive Learning of Visual Representations' by Chen et al. (2020), which introduces SimCLR."),
    HumanMessage(content="How does that method handle negative samples?"),
]

MULTI_TURN_MESSAGES = [
    HumanMessage(content="Can you find papers about retrieval-augmented generation?"),
    AIMessage(content="I found 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks' by Lewis et al. (2020)."),
    HumanMessage(content="How does their approach retrieve relevant documents?"),
]

ABOVE_REF_MESSAGES = [
    HumanMessage(content="Find me the LoRA paper on efficient fine-tuning of large language models."),
    AIMessage(content="I found 'LoRA: Low-Rank Adaptation of Large Language Models' by Hu et al. (2021)."),
    HumanMessage(content="What are the limitations of the above paper?"),
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(kw.lower() in text.lower() for kw in keywords)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_search_resolves_pronoun_reference():
    """'it' should be resolved to 'Transformer' or 'attention'."""
    t.log_inputs({"messages": [m.content for m in PRONOUN_REF_MESSAGES]})

    result = _optimize_for_search(PRONOUN_REF_MESSAGES)
    t.log_outputs({"search_query": result})
    print(f"\n[search] pronoun ref -> {result!r}")

    resolved = _contains_any(result, ["transformer", "attention", "vaswani"])
    t.log_feedback(key="reference_resolved", score=int(resolved))

    assert len(result) > 0
    assert resolved, f"Expected Transformer/attention/Vaswani, got: {result!r}"
    assert "it use" not in result.lower(), f"Unresolved pronoun 'it' still present: {result!r}"


@pytest.mark.langsmith
def test_search_resolves_method_reference():
    """'that method' should be resolved to SimCLR or contrastive learning."""
    t.log_inputs({"messages": [m.content for m in METHOD_REF_MESSAGES]})

    result = _optimize_for_search(METHOD_REF_MESSAGES)
    t.log_outputs({"search_query": result})
    print(f"\n[search] method ref -> {result!r}")

    resolved = _contains_any(result, ["simclr", "contrastive", "negative samples", "chen"])
    t.log_feedback(key="reference_resolved", score=int(resolved))

    assert len(result) > 0
    assert resolved, f"Expected SimCLR/contrastive/negative samples, got: {result!r}"


@pytest.mark.langsmith
def test_search_resolves_their_approach():
    """'their approach' should be resolved to RAG or Lewis et al."""
    t.log_inputs({"messages": [m.content for m in MULTI_TURN_MESSAGES]})

    result = _optimize_for_search(MULTI_TURN_MESSAGES)
    t.log_outputs({"search_query": result})
    print(f"\n[search] their approach -> {result!r}")

    resolved = _contains_any(result, ["retrieval", "rag", "augmented generation", "lewis"])
    t.log_feedback(key="reference_resolved", score=int(resolved))

    assert len(result) > 0
    assert resolved, f"Expected retrieval/RAG, got: {result!r}"


@pytest.mark.langsmith
def test_search_resolves_above_paper():
    """'the above paper' should be resolved to LoRA."""
    t.log_inputs({"messages": [m.content for m in ABOVE_REF_MESSAGES]})

    result = _optimize_for_search(ABOVE_REF_MESSAGES)
    t.log_outputs({"search_query": result})
    print(f"\n[search] above paper -> {result!r}")

    resolved = _contains_any(result, ["lora", "low-rank", "hu"])
    t.log_feedback(key="reference_resolved", score=int(resolved))

    assert len(result) > 0
    assert resolved, f"Expected LoRA/low-rank/Hu, got: {result!r}"


@pytest.mark.langsmith
def test_search_is_keyword_style():
    """Output should be keyword phrases, not a full question."""
    t.log_inputs({"messages": [m.content for m in PRONOUN_REF_MESSAGES]})

    result = _optimize_for_search(PRONOUN_REF_MESSAGES)
    t.log_outputs({"search_query": result})
    print(f"\n[search] keyword style -> {result!r}")

    word_count = len(result.split())
    t.log_feedback(key="is_keyword_style", score=int("?" not in result and word_count <= 20))
    t.log_feedback(key="word_count", score=word_count)

    assert "?" not in result, f"Should not be a question: {result!r}"
    assert word_count <= 20, f"Too long ({word_count} words): {result!r}"
