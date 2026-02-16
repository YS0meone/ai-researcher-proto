"""
Integration tests for the replanner node.

The replanner has two distinct phases separated by interrupt():

  Phase 1 — INTERRUPT FIRES
    replanner() is called for the first time after find_papers completed.
    interrupt("select_papers") raises GraphInterrupt, pausing the graph.
    Nothing after that line runs yet.

  Phase 2 — RESUME
    LangGraph re-runs replanner() from the top.  interrupt() now returns the
    resume payload immediately instead of raising.  The replanner extracts
    selected_paper_ids and an optional user message from the payload, calls
    the LLM to decide remaining plan steps, then returns updated state.

Test coverage:
  • interrupt fires with the correct value
  • resume payload extraction (dict with ids+message, dict without message,
    plain string fallback)
  • selected_paper_ids are committed to state so the QA tool can read them
  • user message is appended to state messages when provided
  • LLM replan decisions: keep QA, drop QA, re-search, done
  • multi-turn conversation context influences replan
  • ambiguous / subtle post-interrupt user messages

Run with:
    cd backend && uv run pytest tests/agent/test_replanner.py -v -s
    LANGSMITH_TEST_SUITE="Replanner" uv run pytest tests/agent/test_replanner.py -v -s
"""

import pytest
from unittest.mock import patch, call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import testing as t
from langgraph.errors import GraphInterrupt

from app.agent.graph import replanner
from app.core.schema import S2Paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_paper(paper_id: str, title: str, abstract: str = "An abstract.") -> S2Paper:
    return S2Paper(paperId=paper_id, title=title, abstract=abstract)


PAPERS = [
    _stub_paper("p1", "Attention Is All You Need",
                "The Transformer model using self-attention for NLP tasks."),
    _stub_paper("p2", "BERT: Pre-training of Deep Bidirectional Transformers",
                "Bidirectional pre-training for language representation."),
    _stub_paper("p3", "GPT-4 Technical Report",
                "Large multimodal model with strong reasoning capabilities."),
]


def _make_state(
    messages: list,
    papers: list | None = None,
    plan_steps: list | None = None,
    selected_paper_ids: list | None = None,
) -> dict:
    import uuid
    ui_msg = AIMessage(id=str(uuid.uuid4()), content="")
    return {
        "messages": messages,
        "papers": papers if papers is not None else PAPERS,
        "plan_steps": plan_steps if plan_steps is not None else [],
        "selected_paper_ids": selected_paper_ids or [],
        "steps": [],
        "ui_tracking_message": ui_msg,
        "ui_tracking_id": "test-ui-id",
        "is_clear": True,
    }


def _base_messages():
    """Minimal message history that represents a completed find_papers step."""
    return [
        HumanMessage(content="Find me papers about transformer architectures."),
        AIMessage(content="", tool_calls=[{"name": "find_papers", "args": {}, "id": "tc1", "type": "tool_call"}]),
        ToolMessage(content="I found 3 papers for your query.", tool_call_id="tc1", name="find_papers"),
        AIMessage(content="", id="paper-ui-msg"),  # paper list UI message committed by post_tool
    ]


def _run_replanner_resumed(state: dict, resume_payload) -> dict:
    """Simulate the resumed execution: interrupt() returns resume_payload immediately."""
    with patch("app.agent.ui_manager.push_ui_message"):
        with patch("app.agent.graph.interrupt", return_value=resume_payload):
            return replanner(state)


# ---------------------------------------------------------------------------
# Phase 1 — interrupt fires
# ---------------------------------------------------------------------------

def test_interrupt_fires_with_correct_value():
    """On first call, interrupt() must be invoked with 'select_papers'."""
    # Real interrupt() requires a LangGraph runnable context (get_config()),
    # which doesn't exist in unit tests.  Mock it to raise GraphInterrupt so
    # we can verify the call argument without needing a live graph runner.
    state = _make_state(_base_messages())

    def fake_interrupt(value):
        raise GraphInterrupt(value)

    with patch("app.agent.ui_manager.push_ui_message"):
        with patch("app.agent.graph.interrupt", side_effect=fake_interrupt) as mock_interrupt:
            with pytest.raises(GraphInterrupt):
                replanner(state)

    mock_interrupt.assert_called_once_with("select_papers")


def test_interrupt_called_before_llm():
    """interrupt() must be called before the replanner LLM is ever invoked."""
    state = _make_state(_base_messages())

    llm_invoked = []

    def fake_interrupt(value):
        raise GraphInterrupt(value)

    with patch("app.agent.ui_manager.push_ui_message"):
        with patch("app.agent.graph.interrupt", side_effect=fake_interrupt) as mock_interrupt:
            with patch("app.agent.graph.supervisor_model") as mock_model:
                mock_model.bind_tools.side_effect = lambda *a, **kw: (_ for _ in ()).throw(
                    AssertionError("LLM was called before interrupt")
                )
                with pytest.raises(GraphInterrupt):
                    replanner(state)
        mock_interrupt.assert_called_once_with("select_papers")
        assert not llm_invoked


# ---------------------------------------------------------------------------
# Phase 2 — resume payload extraction
# ---------------------------------------------------------------------------

def test_resume_dict_payload_extracts_ids_and_message():
    """Dict payload: both selected_paper_ids and user_message are extracted."""
    state = _make_state(_base_messages(), plan_steps=["retrieve_and_answer_question"])
    payload = {"selected_paper_ids": ["p1", "p2"], "user_message": "Explain their methodology."}

    result = _run_replanner_resumed(state, payload)

    assert result["selected_paper_ids"] == ["p1", "p2"]
    assert any(
        isinstance(m, HumanMessage) and "methodology" in m.content.lower()
        for m in result["messages"]
    ), "User message should be appended to state messages"


def test_resume_dict_payload_no_message():
    """Dict payload without user_message: no HumanMessage is added."""
    state = _make_state(_base_messages(), plan_steps=["retrieve_and_answer_question"])
    payload = {"selected_paper_ids": ["p1"], "user_message": None}

    result = _run_replanner_resumed(state, payload)

    assert result["selected_paper_ids"] == ["p1"]
    assert result["messages"] == [], "No new messages should be added when user_message is None"


def test_resume_plain_string_fallback():
    """Plain string resume value falls back to state's selected_paper_ids."""
    state = _make_state(
        _base_messages(),
        plan_steps=["retrieve_and_answer_question"],
        selected_paper_ids=["p3"],
    )

    result = _run_replanner_resumed(state, "continue")

    # Falls back to state value
    assert result["selected_paper_ids"] == ["p3"]
    # "continue" is a sentinel — should NOT be added as a human message
    assert result["messages"] == []


def test_resume_empty_string_fallback():
    """Empty string resume: same fallback as 'continue', no human message added."""
    state = _make_state(
        _base_messages(),
        plan_steps=[],
        selected_paper_ids=["p1"],
    )
    result = _run_replanner_resumed(state, "")
    assert result["messages"] == []


# ---------------------------------------------------------------------------
# selected_paper_ids committed to state
# ---------------------------------------------------------------------------

def test_selected_paper_ids_always_in_return():
    """selected_paper_ids must always be present in the return dict (for QA tool)."""
    state = _make_state(_base_messages(), plan_steps=[])
    payload = {"selected_paper_ids": ["p2"], "user_message": None}

    result = _run_replanner_resumed(state, payload)

    assert "selected_paper_ids" in result
    assert result["selected_paper_ids"] == ["p2"]


def test_empty_selection_still_committed():
    """Even with no papers selected, selected_paper_ids=[] is committed."""
    state = _make_state(_base_messages(), plan_steps=["retrieve_and_answer_question"])
    payload = {"selected_paper_ids": [], "user_message": None}

    result = _run_replanner_resumed(state, payload)

    assert "selected_paper_ids" in result
    assert result["selected_paper_ids"] == []


# ---------------------------------------------------------------------------
# LLM replan decisions
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_replan_keeps_qa_when_papers_selected():
    """Papers selected + original plan included QA -> replanner keeps retrieve_and_answer_question.

    The original question is visible in the message history so the replanner
    knows the user still wants an answer, not just to browse papers.
    """
    messages = [
        HumanMessage(content="Find papers on transformer architectures and explain their self-attention mechanism."),
        AIMessage(content="", tool_calls=[{"name": "find_papers", "args": {}, "id": "tc1", "type": "tool_call"}]),
        ToolMessage(content="I found 3 papers.", tool_call_id="tc1", name="find_papers"),
        AIMessage(content="", id="paper-ui-msg"),
    ]
    state = _make_state(messages, plan_steps=["retrieve_and_answer_question"])
    payload = {
        "selected_paper_ids": ["p1", "p2"],
        "user_message": "Yes, these look good.",
    }
    t.log_inputs({"plan_steps": state["plan_steps"], "payload": payload})

    result = _run_replanner_resumed(state, payload)
    t.log_outputs({"plan_steps": result["plan_steps"]})
    print(f"\n[replanner] keep qa -> {result['plan_steps']}")

    t.log_feedback(key="correct_plan", score=int(result["plan_steps"] == ["retrieve_and_answer_question"]))
    assert result["plan_steps"] == ["retrieve_and_answer_question"]


@pytest.mark.langsmith
def test_replan_drops_qa_when_user_satisfied_no_question():
    """User says they're done browsing, no QA needed -> empty plan."""
    state = _make_state(
        _base_messages(),
        plan_steps=[],  # find_only was the original plan
    )
    payload = {
        "selected_paper_ids": ["p1"],
        "user_message": "Thanks, that's all I needed.",
    }
    t.log_inputs({"plan_steps": state["plan_steps"], "payload": payload})

    result = _run_replanner_resumed(state, payload)
    t.log_outputs({"plan_steps": result["plan_steps"]})
    print(f"\n[replanner] drop qa -> {result['plan_steps']}")

    t.log_feedback(key="correct_plan", score=int(result["plan_steps"] == []))
    assert result["plan_steps"] == []


@pytest.mark.langsmith
def test_replan_triggers_new_search_on_dissatisfaction():
    """User says results are wrong and wants a different search -> find_papers."""
    state = _make_state(
        _base_messages(),
        plan_steps=[],
    )
    payload = {
        "selected_paper_ids": [],
        "user_message": "These aren't what I wanted. Search for papers on BERT instead.",
    }
    t.log_inputs({"plan_steps": state["plan_steps"], "payload": payload})

    result = _run_replanner_resumed(state, payload)
    t.log_outputs({"plan_steps": result["plan_steps"]})
    print(f"\n[replanner] re-search -> {result['plan_steps']}")

    assert "find_papers" in result["plan_steps"]
    t.log_feedback(key="correct_plan", score=int("find_papers" in result["plan_steps"]))


@pytest.mark.langsmith
def test_replan_triggers_find_then_qa_on_new_search_with_question():
    """User wants different papers AND still has a question -> find_papers + QA."""
    state = _make_state(
        _base_messages(),
        plan_steps=["retrieve_and_answer_question"],
    )
    payload = {
        "selected_paper_ids": [],
        "user_message": "These are not relevant. Find papers on LoRA instead and explain how it works.",
    }
    t.log_inputs({"plan_steps": state["plan_steps"], "payload": payload})

    result = _run_replanner_resumed(state, payload)
    t.log_outputs({"plan_steps": result["plan_steps"]})
    print(f"\n[replanner] re-search+qa -> {result['plan_steps']}")

    expected = ["find_papers", "retrieve_and_answer_question"]
    t.log_feedback(key="correct_plan", score=int(result["plan_steps"] == expected))
    assert result["plan_steps"] == expected


@pytest.mark.langsmith
def test_replan_no_selection_no_message_empties_plan():
    """User clicks Continue with no papers selected and no message -> done."""
    state = _make_state(
        _base_messages(),
        plan_steps=[],
    )
    payload = {"selected_paper_ids": [], "user_message": None}
    t.log_inputs({"plan_steps": state["plan_steps"], "payload": payload})

    result = _run_replanner_resumed(state, payload)
    t.log_outputs({"plan_steps": result["plan_steps"]})
    print(f"\n[replanner] no selection no message -> {result['plan_steps']}")

    t.log_feedback(key="correct_plan", score=int(result["plan_steps"] == []))
    assert result["plan_steps"] == []


# ---------------------------------------------------------------------------
# Multi-turn conversation context
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_replan_multiturn_original_qa_intent_preserved():
    """Mid-session: user asked find+summarize, found papers, selects them -> QA preserved."""
    messages = [
        HumanMessage(content="Find papers about mixture of experts and summarize their routing strategies."),
        AIMessage(content="", tool_calls=[{"name": "find_papers", "args": {}, "id": "tc1", "type": "tool_call"}]),
        ToolMessage(content="I found 3 papers.", tool_call_id="tc1", name="find_papers"),
        AIMessage(content="", id="paper-ui-msg"),
    ]
    state = _make_state(messages, plan_steps=["retrieve_and_answer_question"])
    payload = {
        "selected_paper_ids": ["p1", "p2"],
        "user_message": "These look right, go ahead.",
    }
    t.log_inputs({"messages": [m.content for m in messages if hasattr(m, "content") and m.content], "payload": payload})

    result = _run_replanner_resumed(state, payload)
    t.log_outputs({"plan_steps": result["plan_steps"]})
    print(f"\n[replanner] multiturn keep qa -> {result['plan_steps']}")

    t.log_feedback(key="correct_plan", score=int(result["plan_steps"] == ["retrieve_and_answer_question"]))
    assert result["plan_steps"] == ["retrieve_and_answer_question"]


@pytest.mark.langsmith
def test_replan_multiturn_pivot_to_different_topic():
    """After reviewing papers, user pivots to a completely different topic -> find_papers."""
    messages = [
        HumanMessage(content="Find papers on attention mechanisms."),
        AIMessage(content="", tool_calls=[{"name": "find_papers", "args": {}, "id": "tc1", "type": "tool_call"}]),
        ToolMessage(content="I found 3 papers.", tool_call_id="tc1", name="find_papers"),
        AIMessage(content="", id="paper-ui-msg"),
    ]
    state = _make_state(messages, plan_steps=[])
    payload = {
        "selected_paper_ids": [],
        "user_message": "Actually, can you find papers on positional encodings instead?",
    }
    t.log_inputs({"payload": payload})

    result = _run_replanner_resumed(state, payload)
    t.log_outputs({"plan_steps": result["plan_steps"]})
    print(f"\n[replanner] multiturn pivot -> {result['plan_steps']}")

    assert "find_papers" in result["plan_steps"]
    t.log_feedback(key="correct_plan", score=int("find_papers" in result["plan_steps"]))


# ---------------------------------------------------------------------------
# Subtle / ambiguous post-interrupt messages
# ---------------------------------------------------------------------------

@pytest.mark.langsmith
def test_replan_subtle_implicit_proceed():
    """'Perfect' with papers selected, QA in plan, and original question visible -> keep QA."""
    messages = [
        HumanMessage(content="Find papers on BERT and summarize what pre-training tasks they use."),
        AIMessage(content="", tool_calls=[{"name": "find_papers", "args": {}, "id": "tc1", "type": "tool_call"}]),
        ToolMessage(content="I found 3 papers.", tool_call_id="tc1", name="find_papers"),
        AIMessage(content="", id="paper-ui-msg"),
    ]
    state = _make_state(messages, plan_steps=["retrieve_and_answer_question"])
    payload = {
        "selected_paper_ids": ["p1"],
        "user_message": "Perfect.",
    }
    t.log_inputs({"payload": payload, "remaining": state["plan_steps"]})

    result = _run_replanner_resumed(state, payload)
    t.log_outputs({"plan_steps": result["plan_steps"]})
    print(f"\n[replanner] subtle proceed -> {result['plan_steps']}")

    t.log_feedback(key="correct_plan", score=int(result["plan_steps"] == ["retrieve_and_answer_question"]))
    assert result["plan_steps"] == ["retrieve_and_answer_question"]


@pytest.mark.langsmith
def test_replan_subtle_implicit_done_no_qa():
    """'Great, thanks' with no QA in remaining plan -> empty (done)."""
    state = _make_state(_base_messages(), plan_steps=[])
    payload = {
        "selected_paper_ids": ["p1"],
        "user_message": "Great, thanks.",
    }
    t.log_inputs({"payload": payload, "remaining": state["plan_steps"]})

    result = _run_replanner_resumed(state, payload)
    t.log_outputs({"plan_steps": result["plan_steps"]})
    print(f"\n[replanner] subtle done -> {result['plan_steps']}")

    t.log_feedback(key="correct_plan", score=int(result["plan_steps"] == []))
    assert result["plan_steps"] == []


@pytest.mark.langsmith
def test_replan_subtle_more_like_this():
    """'Find more papers like these' -> another find_papers round."""
    state = _make_state(_base_messages(), plan_steps=[])
    payload = {
        "selected_paper_ids": ["p1"],
        "user_message": "Can you find more papers like these?",
    }
    t.log_inputs({"payload": payload})

    result = _run_replanner_resumed(state, payload)
    t.log_outputs({"plan_steps": result["plan_steps"]})
    print(f"\n[replanner] subtle more like this -> {result['plan_steps']}")

    assert "find_papers" in result["plan_steps"]
    t.log_feedback(key="correct_plan", score=int("find_papers" in result["plan_steps"]))


@pytest.mark.langsmith
def test_replan_subtle_explain_selected():
    """'Explain these' with papers selected and no remaining QA step -> add QA."""
    state = _make_state(_base_messages(), plan_steps=[])  # original was find_only
    payload = {
        "selected_paper_ids": ["p1", "p2"],
        "user_message": "Can you explain what these papers are about?",
    }
    t.log_inputs({"payload": payload, "remaining": state["plan_steps"]})

    result = _run_replanner_resumed(state, payload)
    t.log_outputs({"plan_steps": result["plan_steps"]})
    print(f"\n[replanner] subtle explain selected -> {result['plan_steps']}")

    # User now wants QA even though original plan was find_only
    t.log_feedback(key="correct_plan", score=int("retrieve_and_answer_question" in result["plan_steps"]))
    assert "retrieve_and_answer_question" in result["plan_steps"]
