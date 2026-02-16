"""
Integration tests for the planner node.

Tests that the planner correctly maps user intent to one of three plan shapes:
    ["find_papers"]
    ["retrieve_and_answer_question"]
    ["find_papers", "retrieve_and_answer_question"]

Coverage:
  • Unambiguous find-only and qa-only inputs
  • Ambiguous / subtle queries (e.g. "find and summarize", implicit QA intent)
  • Multi-turn conversations
  • Effect of query-clarification context on planner output

Run with:
    cd backend && uv run pytest tests/agent/test_planner.py -v -s
    LANGSMITH_TEST_SUITE="Planner" uv run pytest tests/agent/test_planner.py -v -s
"""

import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import testing as t

from app.agent.graph import planner
from app.core.schema import S2Paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_paper(paper_id: str = "p1", title: str = "Some Paper") -> S2Paper:
    return S2Paper(paperId=paper_id, title=title, abstract="An abstract.")


def _make_state(
    messages: list,
    papers: list | None = None,
    selected_paper_ids: list | None = None,
    steps: list | None = None,
    ui_tracking_message=None,
    ui_tracking_id: str = "test-ui-id",
) -> dict:
    """Build a minimal SupervisorState dict for the planner node."""
    import uuid
    if ui_tracking_message is None:
        ui_tracking_message = AIMessage(id=str(uuid.uuid4()), content="")
    return {
        "messages": messages,
        "papers": papers or [],
        "selected_paper_ids": selected_paper_ids or [],
        "steps": steps or [],
        "ui_tracking_message": ui_tracking_message,
        "ui_tracking_id": ui_tracking_id,
        "plan_steps": [],
        "is_clear": True,
    }


def _run_planner(messages, papers=None, selected_paper_ids=None) -> list[str]:
    state = _make_state(messages, papers=papers, selected_paper_ids=selected_paper_ids)
    # push_ui_message requires a live LangGraph runnable context; mock it out.
    with patch("app.agent.ui_manager.push_ui_message"):
        result = planner(state)
    return result["plan_steps"]


# ---------------------------------------------------------------------------
# Pattern 1 -unambiguous find-only
# ---------------------------------------------------------------------------

FIND_ONLY_CASES = [
    (
        "bare find request",
        [HumanMessage(content="Find me papers about transformer architectures.")],
        ["find_papers"],
    ),
    (
        "collect / retrieve phrasing",
        [HumanMessage(content="Can you search for recent papers on diffusion models?")],
        ["find_papers"],
    ),
    (
        "list request",
        [HumanMessage(content="Give me a list of papers on reinforcement learning from human feedback.")],
        ["find_papers"],
    ),
    (
        "find with author name",
        [HumanMessage(content="Find the LoRA paper by Hu et al.")],
        ["find_papers"],
    ),
]

@pytest.mark.parametrize("label,messages,expected", FIND_ONLY_CASES, ids=[c[0] for c in FIND_ONLY_CASES])
@pytest.mark.langsmith
def test_find_only(label, messages, expected):
    """Planner should produce ['find_papers'] for pure retrieval requests."""
    t.log_inputs({"label": label, "messages": [m.content for m in messages]})

    result = _run_planner(messages)
    t.log_outputs({"plan_steps": result})
    print(f"\n[planner] {label} ->{result}")

    t.log_feedback(key="correct_plan", score=int(result == expected))
    assert result == expected, f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# Pattern 2 -unambiguous qa-only (papers already present)
# ---------------------------------------------------------------------------

EXISTING_PAPERS = [_stub_paper("p1", "Attention Is All You Need")]
SELECTED_IDS = ["p1"]

QA_ONLY_CASES = [
    (
        "direct question about loaded paper",
        [
            HumanMessage(content="Find papers about the Transformer architecture."),
            AIMessage(content="I found 'Attention Is All You Need' by Vaswani et al. and added it to your list."),
            HumanMessage(content="What attention mechanism does it use?"),
        ],
        EXISTING_PAPERS,
        SELECTED_IDS,
    ),
    (
        "explain paper in context",
        [
            HumanMessage(content="Can you explain the methodology of the paper we just loaded?"),
        ],
        EXISTING_PAPERS,
        SELECTED_IDS,
    ),
    (
        "compare papers already present",
        [
            HumanMessage(content="Compare the two papers I selected."),
        ],
        [_stub_paper("p1", "Paper A"), _stub_paper("p2", "Paper B")],
        ["p1", "p2"],
    ),
]

@pytest.mark.parametrize("label,messages,papers,sel_ids", QA_ONLY_CASES, ids=[c[0] for c in QA_ONLY_CASES])
@pytest.mark.langsmith
def test_qa_only(label, messages, papers, sel_ids):
    """Planner should produce ['retrieve_and_answer_question'] when papers are present and user asks a question."""
    t.log_inputs({"label": label, "messages": [m.content for m in messages], "num_papers": len(papers)})

    result = _run_planner(messages, papers=papers, selected_paper_ids=sel_ids)
    t.log_outputs({"plan_steps": result})
    print(f"\n[planner] {label} ->{result}")

    expected = ["retrieve_and_answer_question"]
    t.log_feedback(key="correct_plan", score=int(result == expected))
    assert result == expected, f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# Pattern 3 -find then QA (subtle / ambiguous intent)
# ---------------------------------------------------------------------------
# These are the trickiest cases: the user asks to *find* papers and *also*
# wants something done with them (summarise, explain, compare, etc.).

FIND_THEN_QA_CASES = [
    (
        "find and summarize",
        [HumanMessage(content="Find me papers about RAG and summarize their main contributions.")],
    ),
    (
        "find and explain",
        [HumanMessage(content="Find the LoRA paper and explain how it reduces trainable parameters.")],
    ),
    (
        "find and tell me about",
        [HumanMessage(content="Can you find papers on contrastive learning and tell me what datasets they use?")],
    ),
    (
        "implicit find + question -what does it say",
        [HumanMessage(content="Look up the GPT-4 technical report and tell me what it says about safety evaluations.")],
    ),
    (
        "find + compare -comparative implicit QA",
        [HumanMessage(content="Find papers on knowledge distillation and compare their approaches.")],
    ),
    (
        "ambiguous overview request",
        [HumanMessage(content="Get me some papers on chain-of-thought prompting and give me an overview.")],
    ),
    (
        "subtle: 'and let me know' phrasing",
        [HumanMessage(content="Search for papers on speculative decoding and let me know what the key ideas are.")],
    ),
]

@pytest.mark.parametrize("label,messages", FIND_THEN_QA_CASES, ids=[c[0] for c in FIND_THEN_QA_CASES])
@pytest.mark.langsmith
def test_find_then_qa(label, messages):
    """Planner should produce ['find_papers', 'retrieve_and_answer_question'] when the query implies both retrieval and answering."""
    t.log_inputs({"label": label, "messages": [m.content for m in messages]})

    result = _run_planner(messages, papers=[])  # no papers yet
    t.log_outputs({"plan_steps": result})
    print(f"\n[planner] {label} ->{result}")

    expected = ["find_papers", "retrieve_and_answer_question"]
    t.log_feedback(key="correct_plan", score=int(result == expected))
    assert result == expected, f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# Multi-turn conversations
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_multiturn_find_after_qa_session():
    """After a QA session the user asks to find *different* papers ->find_only."""
    messages = [
        HumanMessage(content="Find me papers about BERT."),
        AIMessage(content="I found several papers on BERT."),
        HumanMessage(content="What pre-training tasks does BERT use?"),
        AIMessage(content="BERT uses masked language modelling and next-sentence prediction."),
        HumanMessage(content="Now find me papers about GPT-2 instead."),
    ]
    t.log_inputs({"messages": [m.content for m in messages]})

    result = _run_planner(messages, papers=[_stub_paper("p1", "BERT Paper")])
    t.log_outputs({"plan_steps": result})
    print(f"\n[planner] multiturn find after qa ->{result}")

    expected = ["find_papers"]
    t.log_feedback(key="correct_plan", score=int(result == expected))
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.langsmith
def test_multiturn_follow_up_question_with_papers():
    """User asks a follow-up question in a thread where papers are already loaded ->qa_only."""
    messages = [
        HumanMessage(content="Find papers on mixture of experts."),
        AIMessage(content="I found papers on Mixture of Experts (MoE)."),
        HumanMessage(content="What are the routing strategies discussed?"),
    ]
    papers = [_stub_paper("p1", "Outrageously Large Neural Networks"), _stub_paper("p2", "Switch Transformers")]
    t.log_inputs({"messages": [m.content for m in messages], "num_papers": len(papers)})

    result = _run_planner(messages, papers=papers, selected_paper_ids=["p1", "p2"])
    t.log_outputs({"plan_steps": result})
    print(f"\n[planner] multiturn follow-up question ->{result}")

    expected = ["retrieve_and_answer_question"]
    t.log_feedback(key="correct_plan", score=int(result == expected))
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.langsmith
def test_multiturn_find_then_qa_after_chat_history():
    """Midway through a session the user asks to find new papers AND answer something ->find_then_qa."""
    messages = [
        HumanMessage(content="Find me papers about attention mechanisms."),
        AIMessage(content="Found some papers."),
        HumanMessage(content="Now find papers on positional encodings and explain the difference between absolute and relative ones."),
    ]
    papers = [_stub_paper("p1", "Attention Is All You Need")]
    t.log_inputs({"messages": [m.content for m in messages]})

    # existing papers are about attention, not positional encodings ->planner should find new ones
    result = _run_planner(messages, papers=papers)
    t.log_outputs({"plan_steps": result})
    print(f"\n[planner] multiturn find_then_qa ->{result}")

    expected = ["find_papers", "retrieve_and_answer_question"]
    t.log_feedback(key="correct_plan", score=int(result == expected))
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.langsmith
def test_multiturn_pronoun_resolution_find_only():
    """'Find more like it' -pronoun should not cause planner to add a QA step."""
    messages = [
        HumanMessage(content="Find the Flash Attention paper."),
        AIMessage(content="I found 'FlashAttention: Fast and Memory-Efficient Exact Attention' by Dao et al."),
        HumanMessage(content="Find more papers like it."),
    ]
    t.log_inputs({"messages": [m.content for m in messages]})

    result = _run_planner(messages, papers=[_stub_paper("p1", "FlashAttention")])
    t.log_outputs({"plan_steps": result})
    print(f"\n[planner] pronoun resolution find only ->{result}")

    expected = ["find_papers"]
    t.log_feedback(key="correct_plan", score=int(result == expected))
    assert result == expected, f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# Effect of query-clarification context on planner
# ---------------------------------------------------------------------------
# query_clarification runs before planner and may add a SystemMessage or
# AIMessage clarifying the intent.  We simulate that by prepending a
# clarification exchange to the message history.

@pytest.mark.langsmith
def test_clarification_resolves_ambiguous_to_find_then_qa():
    """An ambiguous original query becomes find_then_qa after clarification."""
    # Simulated output of query_clarification that asked the user to be specific
    messages = [
        HumanMessage(content="Show me stuff about LLMs."),
        AIMessage(content="Could you clarify: are you looking for papers to browse, or do you have a specific question you'd like answered?"),
        HumanMessage(content="I want to find papers on LLM alignment and then understand their evaluation methods."),
    ]
    t.log_inputs({"messages": [m.content for m in messages]})

    result = _run_planner(messages)
    t.log_outputs({"plan_steps": result})
    print(f"\n[planner] clarification ->find_then_qa: {result}")

    expected = ["find_papers", "retrieve_and_answer_question"]
    t.log_feedback(key="correct_plan", score=int(result == expected))
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.langsmith
def test_clarification_resolves_ambiguous_to_find_only():
    """After clarification the user just wants to browse papers, not ask a question."""
    messages = [
        HumanMessage(content="Something about neural networks."),
        AIMessage(content="Are you looking for papers to read, or do you have a specific question?"),
        HumanMessage(content="Just find me some good recent papers on graph neural networks."),
    ]
    t.log_inputs({"messages": [m.content for m in messages]})

    result = _run_planner(messages)
    t.log_outputs({"plan_steps": result})
    print(f"\n[planner] clarification ->find_only: {result}")

    expected = ["find_papers"]
    t.log_feedback(key="correct_plan", score=int(result == expected))
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.langsmith
def test_clarification_resolves_ambiguous_to_qa_only():
    """After clarification the user only wants to ask about papers already loaded."""
    papers = [_stub_paper("p1", "RLHF Paper")]
    messages = [
        HumanMessage(content="Tell me about reinforcement learning."),
        AIMessage(content="Could you clarify: do you want me to search for new papers, or answer a question about the papers already loaded?"),
        HumanMessage(content="I want to ask a question about the RLHF paper that's already there."),
    ]
    t.log_inputs({"messages": [m.content for m in messages], "num_papers": len(papers)})

    result = _run_planner(messages, papers=papers, selected_paper_ids=["p1"])
    t.log_outputs({"plan_steps": result})
    print(f"\n[planner] clarification ->qa_only: {result}")

    expected = ["retrieve_and_answer_question"]
    t.log_feedback(key="correct_plan", score=int(result == expected))
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.langsmith
def test_clarification_then_find_then_qa_subtle():
    """After clarification, a subtle 'find and understand' phrasing ->find_then_qa."""
    messages = [
        HumanMessage(content="Help me with some AI safety reading."),
        AIMessage(content="Happy to help! Should I search for papers, or do you have a specific question about papers you've already loaded?"),
        HumanMessage(content="Search for papers on constitutional AI and help me understand the key principles they propose."),
    ]
    t.log_inputs({"messages": [m.content for m in messages]})

    result = _run_planner(messages)
    t.log_outputs({"plan_steps": result})
    print(f"\n[planner] clarification subtle find_then_qa: {result}")

    expected = ["find_papers", "retrieve_and_answer_question"]
    t.log_feedback(key="correct_plan", score=int(result == expected))
    assert result == expected, f"Expected {expected}, got {result}"
