"""
Integration tests for _optimize_for_qa.

Verifies that self-contained research questions are correctly derived from
multi-turn conversations, including:
  - Unresolved pronoun/reference resolution from a single follow-up
  - Follow-up questions based on the AI's own previous answers

Run with:
    cd backend && uv run pytest tests/agent/test_optimize_for_qa.py -v -s
    LANGSMITH_TEST_SUITE="Query Optimization - QA" uv run pytest tests/agent/test_optimize_for_qa.py -v -s
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import testing as t

from app.agent.graph import _optimize_for_qa


# ---------------------------------------------------------------------------
# Scenario 1: Simple reference resolution (single follow-up)
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

ABOVE_REF_MESSAGES = [
    HumanMessage(content="Find me the LoRA paper on efficient fine-tuning of large language models."),
    AIMessage(content="I found 'LoRA: Low-Rank Adaptation of Large Language Models' by Hu et al. (2021)."),
    HumanMessage(content="What are the limitations of the above paper?"),
]

THEIR_APPROACH_MESSAGES = [
    HumanMessage(content="Can you find papers about retrieval-augmented generation?"),
    AIMessage(content="I found 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks' by Lewis et al. (2020)."),
    HumanMessage(content="How does their approach retrieve relevant documents?"),
]

# ---------------------------------------------------------------------------
# Scenario 2: Follow-up on the AI's answer
# The user asks about something the AI just explained — the optimizer must
# resolve "that" / "this approach" / "the first one" back to the AI's reply.
# ---------------------------------------------------------------------------

# User asks about the attention mechanism, AI explains it, user drills deeper
FOLLOWUP_ON_AI_ANSWER_MESSAGES = [
    HumanMessage(content="What is the attention mechanism used in the Transformer?"),
    AIMessage(content=(
        "The Transformer uses scaled dot-product attention. "
        "Queries, keys and values are projected into a lower-dimensional space, "
        "then attention scores are computed as softmax(QK^T / sqrt(d_k)) * V. "
        "The model also uses multi-head attention, running several attention heads in parallel "
        "and concatenating their outputs."
    )),
    HumanMessage(content="Why do they divide by sqrt(d_k)?"),
]

# User asks about LoRA, AI explains the rank decomposition, user asks about rank choice
FOLLOWUP_ON_LORA_ANSWER_MESSAGES = [
    HumanMessage(content="Can you explain how LoRA fine-tunes large language models?"),
    AIMessage(content=(
        "LoRA (Low-Rank Adaptation) freezes the pretrained model weights and injects "
        "trainable rank-decomposition matrices into each Transformer layer. "
        "For a weight matrix W, it learns two small matrices A and B such that the "
        "update is W + BA, where the rank r of B and A is much smaller than the original dimensions. "
        "This drastically reduces the number of trainable parameters."
    )),
    HumanMessage(content="How do you choose the right rank for those matrices?"),
]

# Multi-turn: user asks about SimCLR training, AI answers, user asks about a specific detail
FOLLOWUP_ON_SIMCLR_MESSAGES = [
    HumanMessage(content="How does SimCLR train its encoder?"),
    AIMessage(content=(
        "SimCLR trains a visual encoder using contrastive learning. "
        "For each image, two augmented views are created. "
        "The encoder maps each view to a representation, then a projection head maps it to "
        "a lower-dimensional space where the NT-Xent contrastive loss is applied. "
        "Positive pairs (same image) are pulled together and negative pairs (different images) are pushed apart."
    )),
    HumanMessage(content="What augmentations does it apply to create those two views?"),
]

# Longer chain: three turns of Q&A, then a follow-up referencing earlier content
LONG_CHAIN_MESSAGES = [
    HumanMessage(content="Find me papers about knowledge distillation."),
    AIMessage(content="I found 'Distilling the Knowledge in a Neural Network' by Hinton et al. (2015)."),
    HumanMessage(content="How does it transfer knowledge from teacher to student?"),
    AIMessage(content=(
        "The teacher model produces soft probability distributions (soft labels) over classes "
        "using a temperature parameter T > 1. The student is trained to match these soft labels "
        "rather than the hard one-hot labels, which provides richer gradient signal about "
        "inter-class similarities learned by the teacher."
    )),
    HumanMessage(content="What is the role of the temperature parameter in that process?"),
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(kw.lower() in text.lower() for kw in keywords)


# ---------------------------------------------------------------------------
# Tests: reference resolution (single follow-up)
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_qa_resolves_pronoun_reference():
    """'it' should be expanded to the Transformer paper."""
    t.log_inputs({"messages": [m.content for m in PRONOUN_REF_MESSAGES]})

    result = _optimize_for_qa(PRONOUN_REF_MESSAGES)
    t.log_outputs({"qa_query": result})
    print(f"\n[qa] pronoun ref -> {result!r}")

    resolved = _contains_any(result, ["transformer", "attention", "vaswani"])
    no_pronoun = "it" not in result.lower().split()
    t.log_feedback(key="reference_resolved", score=int(resolved))
    t.log_feedback(key="pronoun_eliminated", score=int(no_pronoun))

    assert len(result) > 0
    assert resolved, f"Expected Transformer/attention/Vaswani, got: {result!r}"
    assert no_pronoun, f"Unresolved pronoun 'it' still present: {result!r}"


@pytest.mark.langsmith
def test_qa_resolves_method_reference():
    """'that method' should be expanded to SimCLR."""
    t.log_inputs({"messages": [m.content for m in METHOD_REF_MESSAGES]})

    result = _optimize_for_qa(METHOD_REF_MESSAGES)
    t.log_outputs({"qa_query": result})
    print(f"\n[qa] method ref -> {result!r}")

    resolved = _contains_any(result, ["simclr", "contrastive", "negative"])
    t.log_feedback(key="reference_resolved", score=int(resolved))

    assert len(result) > 0
    assert resolved, f"Expected SimCLR/contrastive/negative, got: {result!r}"


@pytest.mark.langsmith
def test_qa_resolves_above_paper():
    """'the above paper' should be replaced with LoRA explicitly."""
    t.log_inputs({"messages": [m.content for m in ABOVE_REF_MESSAGES]})

    result = _optimize_for_qa(ABOVE_REF_MESSAGES)
    t.log_outputs({"qa_query": result})
    print(f"\n[qa] above paper -> {result!r}")

    resolved = _contains_any(result, ["lora", "low-rank", "hu et al"])
    no_vague = "above" not in result.lower()
    t.log_feedback(key="reference_resolved", score=int(resolved))
    t.log_feedback(key="vague_reference_eliminated", score=int(no_vague))

    assert len(result) > 0
    assert resolved, f"Expected LoRA/low-rank/Hu et al., got: {result!r}"
    assert no_vague, f"Vague reference 'above' still present: {result!r}"


@pytest.mark.langsmith
def test_qa_resolves_their_approach():
    """'their approach' should name RAG / Lewis et al. explicitly."""
    t.log_inputs({"messages": [m.content for m in THEIR_APPROACH_MESSAGES]})

    result = _optimize_for_qa(THEIR_APPROACH_MESSAGES)
    t.log_outputs({"qa_query": result})
    print(f"\n[qa] their approach -> {result!r}")

    resolved = _contains_any(result, ["retrieval-augmented", "retrieval augmented", "rag", "lewis"])
    t.log_feedback(key="reference_resolved", score=int(resolved))

    assert len(result) > 0
    assert resolved, f"Expected RAG/Lewis et al., got: {result!r}"


# ---------------------------------------------------------------------------
# Tests: follow-up on AI's own answer
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_qa_followup_on_attention_explanation():
    """'they' and 'sqrt(d_k)' should be grounded in the Transformer's attention mechanism."""
    t.log_inputs({"messages": [m.content for m in FOLLOWUP_ON_AI_ANSWER_MESSAGES]})

    result = _optimize_for_qa(FOLLOWUP_ON_AI_ANSWER_MESSAGES)
    t.log_outputs({"qa_query": result})
    print(f"\n[qa] followup on AI answer (attention) -> {result!r}")

    resolved = _contains_any(result, ["sqrt", "d_k", "scale", "attention", "transformer", "dot-product"])
    t.log_feedback(key="reference_resolved", score=int(resolved))

    assert len(result) > 0
    assert resolved, f"Expected attention/scaling context, got: {result!r}"


@pytest.mark.langsmith
def test_qa_followup_on_lora_explanation():
    """'those matrices' should be grounded in LoRA's rank-decomposition matrices A and B."""
    t.log_inputs({"messages": [m.content for m in FOLLOWUP_ON_LORA_ANSWER_MESSAGES]})

    result = _optimize_for_qa(FOLLOWUP_ON_LORA_ANSWER_MESSAGES)
    t.log_outputs({"qa_query": result})
    print(f"\n[qa] followup on AI answer (LoRA) -> {result!r}")

    resolved = _contains_any(result, ["lora", "rank", "low-rank", "matrix", "matrices"])
    t.log_feedback(key="reference_resolved", score=int(resolved))

    assert len(result) > 0
    assert resolved, f"Expected LoRA/rank/matrix context, got: {result!r}"


@pytest.mark.langsmith
def test_qa_followup_on_simclr_explanation():
    """'those two views' should be grounded in SimCLR's augmentation strategy."""
    t.log_inputs({"messages": [m.content for m in FOLLOWUP_ON_SIMCLR_MESSAGES]})

    result = _optimize_for_qa(FOLLOWUP_ON_SIMCLR_MESSAGES)
    t.log_outputs({"qa_query": result})
    print(f"\n[qa] followup on AI answer (SimCLR) -> {result!r}")

    resolved = _contains_any(result, ["simclr", "augment", "views", "contrastive"])
    t.log_feedback(key="reference_resolved", score=int(resolved))

    assert len(result) > 0
    assert resolved, f"Expected SimCLR/augmentation context, got: {result!r}"


@pytest.mark.langsmith
def test_qa_followup_on_long_chain():
    """After 3 turns, 'that process' should resolve to knowledge distillation with temperature."""
    t.log_inputs({"messages": [m.content for m in LONG_CHAIN_MESSAGES]})

    result = _optimize_for_qa(LONG_CHAIN_MESSAGES)
    t.log_outputs({"qa_query": result})
    print(f"\n[qa] long chain followup -> {result!r}")

    resolved = _contains_any(result, ["temperature", "distillation", "hinton", "soft", "student", "teacher"])
    t.log_feedback(key="reference_resolved", score=int(resolved))

    assert len(result) > 0
    assert resolved, f"Expected temperature/distillation context, got: {result!r}"


# ---------------------------------------------------------------------------
# Test: output is a complete standalone question
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_qa_output_is_complete_question():
    """QA query should be a full, standalone question regardless of input."""
    t.log_inputs({"messages": [m.content for m in FOLLOWUP_ON_AI_ANSWER_MESSAGES]})

    result = _optimize_for_qa(FOLLOWUP_ON_AI_ANSWER_MESSAGES)
    t.log_outputs({"qa_query": result})
    print(f"\n[qa] complete question check -> {result!r}")

    word_count = len(result.split())
    t.log_feedback(key="word_count", score=word_count)
    t.log_feedback(key="is_question", score=int(result.strip().endswith("?")))

    assert word_count >= 5, f"QA query is too short: {result!r}"
