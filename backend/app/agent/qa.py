"""
QA Agent Module - Agentic RAG for answering questions about selected papers.

This module implements:
1. Vector search within user-selected papers to retrieve relevant segments
2. Question answering grounded in the retrieved evidence
3. Iterative refinement if initial retrieval is insufficient
"""

from typing import Dict, List, Any, Optional
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from app.agent.states import State
from app.agent.utils import get_user_query, setup_langsmith
from app.core.config import settings         
from app.tools.search import vector_search_papers_by_ids_impl
import logging
import sys

logger = logging.getLogger(__name__)
setup_langsmith()
qa_model = init_chat_model(model=settings.MODEL_NAME, api_key=settings.OPENAI_API_KEY)


# ============================================================================
# STRUCTURED OUTPUT MODELS
# ============================================================================

class RetrievalPlan(BaseModel):
    """Structured output for retrieval planning."""
    search_queries: List[str] = Field(
        min_length=1,
        max_length=3,
        description="1-3 focused queries to find relevant evidence"
    )
    reasoning: str = Field(
        description="Why these queries will find relevant evidence")


class AnswerQuality(BaseModel):
    """Structured output for answer quality assessment."""
    is_sufficient: bool = Field(
        description="Whether evidence is sufficient to answer")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the answer")
    missing_info: Optional[str] = Field(
        default=None, description="What info is missing if insufficient")
    refined_query: Optional[str] = Field(
        default=None, description="Refined query if more evidence needed")


# ============================================================================
# QA PROMPTS
# ============================================================================

QA_RETRIEVAL_SYSTEM = """You are a retrieval specialist for academic paper QA.
Your role: Generate focused search queries to find relevant evidence within specific papers.

CONTEXT:
- User has selected specific papers to ask questions about
- You must search within ONLY these papers using vector search
- Goal: Find the most relevant segments to answer the user's question

QUERY GENERATION STRATEGY:
1. Analyze the question - What specific information is needed?
2. Extract key concepts - Technical terms, methods, concepts mentioned
3. Create focused queries - Target the specific information needed
4. Consider variations - Different phrasings that might match paper content

GOOD QUERIES:
- Technical and specific: "attention mechanism computation formula"
- Method-focused: "training procedure hyperparameters"
- Result-focused: "experimental results accuracy benchmark"

BAD QUERIES:
- Too broad: "transformer" (won't find specific info)
- Too narrow: exact phrases that may not match
- Off-topic: queries about things not in the papers

OUTPUT: Generate 1-3 focused search queries."""

QA_ANSWER_SYSTEM = """You are an expert academic QA assistant.
Your role: Answer questions about research papers using retrieved evidence.

CHAIN OF THOUGHT:
1. Review the question and retrieved evidence segments
2. Identify which segments directly address the question
3. Synthesize information from multiple segments if needed
4. Formulate a clear, accurate answer with citations
5. Note any limitations or missing information

ANSWER GUIDELINES:
- Ground EVERY claim in the retrieved evidence
- Cite sources as [Paper: arxiv_id, Section: "quoted text..."]
- If evidence is insufficient, clearly state what's missing
- For technical questions, include specific details (numbers, formulas, etc.)
- Be concise but complete

CITATION FORMAT:
- For direct quotes: "The model achieves 95% accuracy" [Paper: 1234.56789]
- For paraphrased info: According to [Paper: 1234.56789], the approach uses...
- For multiple sources: This finding is supported by [Paper: 1234.56789, Paper: 9876.54321]

HANDLING INSUFFICIENT EVIDENCE:
- If no relevant evidence found: "I couldn't find information about X in the selected papers."
- If partial evidence: "Based on available evidence... However, more details about Y would be needed."
- If conflicting evidence: Present both perspectives with citations."""

QA_QUALITY_SYSTEM = """You are a QA quality assessor.
Your role: Evaluate if the retrieved evidence is sufficient to answer the question.

ASSESSMENT CRITERIA:
1. Relevance - Do segments actually address the question?
2. Completeness - Is enough information available?
3. Specificity - Are technical details present if needed?
4. Clarity - Is the evidence clear enough to formulate an answer?

CONFIDENCE CALIBRATION:
- 0.9-1.0: Evidence directly answers the question with specific details
- 0.7-0.8: Good evidence but some minor gaps
- 0.5-0.6: Partial evidence, can give general answer
- 0.3-0.4: Weak evidence, significant gaps
- 0.0-0.2: No relevant evidence found

OUTPUT: Assess if evidence is sufficient and suggest refinements if needed."""


# ============================================================================
# QA AGENT NODES
# ============================================================================

def qa_prepare(state: State) -> Dict:
    """
    If user hasn't selected specific papers, select top papers from search results.
    """
    selected_ids = state.get("selected_ids", [])
    papers = state.get("papers", [])
    intent = state.get("intent")

    # If QA-only mode and user has selected papers, use them
    if intent == "qa_only" and selected_ids:
        print(
            f"QA mode with {len(selected_ids)} pre-selected papers", file=sys.stderr)
        return {}

    # If search_then_qa mode, select top papers for QA
    if intent == "search_then_qa" and papers and not selected_ids:
        # Select top 5 papers for detailed QA
        top_papers = papers[:5]
        selected_ids = [p["arxiv_id"] for p in top_papers]

        print(
            f"Auto-selecting top {len(selected_ids)} papers for QA", file=sys.stderr)

        return {
            "selected_ids": selected_ids,
            "qa_query": state.get("original_query"),
            "messages": [AIMessage(content=f"Selected top {len(selected_ids)} papers for detailed analysis.")]
        }

    return {}


def qa_retrieve(state: State) -> Dict:
    """
    Retrieve relevant segments from selected papers using vector search.

    Uses the selected_ids from state to scope the vector search to only
    the papers the user has chosen to ask questions about.
    """
    user_msg = get_user_query(state["messages"])
    selected_ids = state.get("selected_ids", [])

    print(
        f"QA Retrieve: query='{user_msg[:50]}...', selected_ids={selected_ids}", file=sys.stderr)

    if not selected_ids:
        print("WARNING: No papers selected for QA!", file=sys.stderr)
        return {
            "retrieved_segments": [],
            "messages": [AIMessage(content="No papers have been selected for Q&A. Please select papers first or use the paper finding mode.")]
        }

    # Generate focused retrieval queries
    retrieval_prompt = f"""User question: {user_msg}

Selected papers to search: {selected_ids}

Generate 1-3 focused search queries to find relevant evidence for this question."""

    structured_model = qa_model.with_structured_output(RetrievalPlan)
    plan = structured_model.invoke([
        SystemMessage(content=QA_RETRIEVAL_SYSTEM),
        HumanMessage(content=retrieval_prompt)
    ])

    print(
        f"Retrieval plan: {plan.search_queries} - {plan.reasoning}", file=sys.stderr)

    # Execute vector searches within selected papers
    all_segments: List[Dict[str, Any]] = []
    seen_content = set()  # Deduplicate by content

    for query in plan.search_queries:
        try:
            results = vector_search_papers_by_ids_impl(
                query=query,
                ids=selected_ids,
                limit=5,
                score_threshold=0.5
            )

            # Add unique results
            for segment in results:
                if "error" not in segment:
                    content_key = segment.get("supporting_detail", "")[:100]
                    if content_key and content_key not in seen_content:
                        seen_content.add(content_key)
                        segment["retrieval_query"] = query
                        all_segments.append(segment)

        except Exception as e:
            print(
                f"Vector search failed for query '{query}': {e}", file=sys.stderr)

    # Sort by similarity score
    all_segments.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

    # Keep top 10 segments
    top_segments = all_segments[:10]

    print(
        f"Retrieved {len(top_segments)} unique segments from {len(selected_ids)} papers", file=sys.stderr)

    return {
        "retrieved_segments": top_segments,
        "qa_query": user_msg
    }


def qa_answer(state: State) -> Dict:
    """
    Generate an answer based on retrieved segments.

    Uses the retrieved_segments to formulate a grounded answer to the user's question.
    Cites sources and acknowledges any gaps in the evidence.
    """
    user_msg = state.get("qa_query") or get_user_query(state["messages"])
    segments = state.get("retrieved_segments", [])
    selected_ids = state.get("selected_ids", [])

    print(
        f"QA Answer: query='{user_msg[:50]}...', segments={len(segments)}", file=sys.stderr)

    if not segments:
        return {
            "messages": [AIMessage(content=f"""I searched the selected papers ({', '.join(selected_ids)}) but couldn't find relevant information to answer your question.

This could mean:
1. The papers don't contain information about this specific topic
2. The question requires details not present in the paper segments indexed
3. The papers might discuss this topic using different terminology

Would you like me to:
- Try a different phrasing of your question?
- Search for different papers that might cover this topic?""")]
        }

    # Format segments for the prompt
    formatted_segments = []
    for i, seg in enumerate(segments, 1):
        formatted_segments.append(f"""
[Segment {i}]
Paper: {seg.get('title', 'Unknown')} (arXiv:{seg.get('arxiv_id', 'unknown')})
Relevance Score: {seg.get('similarity_score', 0):.3f}
Content: {seg.get('supporting_detail', 'No content')}
""")

    segments_text = "\n".join(formatted_segments)

    answer_prompt = f"""User question: {user_msg}

Retrieved evidence from selected papers:
{segments_text}

Provide a comprehensive answer grounded in this evidence. Cite sources for all claims."""

    # Generate answer
    response = qa_model.invoke([
        SystemMessage(content=QA_ANSWER_SYSTEM),
        *state.get("messages", []),
        HumanMessage(content=answer_prompt)
    ])

    return {"messages": [response]}


def qa_assess_quality(state: State) -> str:
    """
    Assess if retrieved evidence is sufficient.

    Returns:
        - "answer" if evidence is sufficient
        - "refine" if more evidence needed
        - "insufficient" if no relevant evidence found
    """
    segments = state.get("retrieved_segments", [])
    user_msg = state.get("qa_query") or get_user_query(state["messages"])

    if not segments:
        return "insufficient"

    # Quick heuristic: if we have high-quality segments, proceed to answer
    high_quality = [s for s in segments if s.get("similarity_score", 0) > 0.7]

    if len(high_quality) >= 2:
        print(
            f"Quality check: {len(high_quality)} high-quality segments found, proceeding to answer", file=sys.stderr)
        return "answer"

    if len(segments) >= 3:
        print(
            f"Quality check: {len(segments)} segments found, proceeding to answer", file=sys.stderr)
        return "answer"

    # More nuanced check with LLM
    assessment_prompt = f"""User question: {user_msg}

Retrieved segments: {len(segments)}
Top segment score: {segments[0].get('similarity_score', 0) if segments else 0}
Top segment preview: {segments[0].get('supporting_detail', '')[:200] if segments else 'None'}

Is this evidence sufficient to answer the question?"""

    try:
        structured_model = qa_model.with_structured_output(AnswerQuality)
        quality = structured_model.invoke([
            SystemMessage(content=QA_QUALITY_SYSTEM),
            HumanMessage(content=assessment_prompt)
        ])

        if quality.is_sufficient or quality.confidence > 0.5:
            return "answer"
        elif quality.refined_query:
            return "refine"
        else:
            return "insufficient"
    except Exception as e:
        print(
            f"Quality assessment failed: {e}, defaulting to answer", file=sys.stderr)
        return "answer"


def qa_refine_retrieval(state: State) -> Dict:
    """
    Refine retrieval with alternative queries when initial results are insufficient.
    """
    user_msg = state.get("qa_query") or get_user_query(state["messages"])
    selected_ids = state.get("selected_ids", [])
    previous_segments = state.get("retrieved_segments", [])

    print(f"Refining retrieval for: {user_msg[:50]}...", file=sys.stderr)

    # Generate alternative queries
    refine_prompt = f"""The initial search didn't find sufficient evidence.

Original question: {user_msg}
Previous results: {len(previous_segments)} segments found
Selected papers: {selected_ids}

Generate 2-3 ALTERNATIVE search queries using different terms or phrasings."""

    structured_model = qa_model.with_structured_output(RetrievalPlan)
    plan = structured_model.invoke([
        SystemMessage(content=QA_RETRIEVAL_SYSTEM),
        HumanMessage(content=refine_prompt)
    ])

    # Execute refined searches
    new_segments: List[Dict[str, Any]] = []
    seen_content = set(seg.get("supporting_detail", "")[
                       :100] for seg in previous_segments)

    for query in plan.search_queries:
        try:
            results = vector_search_papers_by_ids_impl(
                query=query,
                ids=selected_ids,
                limit=5,
                score_threshold=0.4  # Lower threshold for refinement
            )

            for segment in results:
                if "error" not in segment:
                    content_key = segment.get("supporting_detail", "")[:100]
                    if content_key and content_key not in seen_content:
                        seen_content.add(content_key)
                        segment["retrieval_query"] = query
                        new_segments.append(segment)

        except Exception as e:
            print(
                f"Refined search failed for query '{query}': {e}", file=sys.stderr)

    # Merge with previous segments
    all_segments = previous_segments + new_segments
    all_segments.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

    print(
        f"After refinement: {len(all_segments)} total segments", file=sys.stderr)

    return {"retrieved_segments": all_segments[:10]}


# ============================================================================
# QA SUBGRAPH BUILDER
# ============================================================================

def build_qa_graph():
    """
    Build the QA subgraph for paper question answering.

    Flow:
    1. qa_prepare: Prepare state for QA (select papers if needed)
    1. qa_retrieve: Search within selected papers for relevant segments
    2. qa_assess_quality: Check if evidence is sufficient
    3. If insufficient: qa_refine_retrieval (max 1 refinement)
    4. qa_answer: Generate grounded answer
    """
    from langgraph.graph import StateGraph, END

    qa_builder = StateGraph(State)

    # Add nodes
    qa_builder.add_node("qa_prepare", qa_prepare)
    qa_builder.add_node("qa_retrieve", qa_retrieve)
    qa_builder.add_node("qa_assess", lambda s: s)  # Dummy node for conditional
    qa_builder.add_node("qa_refine", qa_refine_retrieval)
    qa_builder.add_node("qa_answer", qa_answer)

    # Add edges
    qa_builder.set_entry_point("qa_prepare")
    qa_builder.add_edge("qa_prepare", "qa_retrieve")
    qa_builder.add_edge("qa_retrieve", "qa_assess")

    # Conditional: assess quality and decide next step
    qa_builder.add_conditional_edges(
        "qa_assess",
        qa_assess_quality,
        {
            "answer": "qa_answer",
            "refine": "qa_refine",
            "insufficient": "qa_answer"  # Still try to answer with available info
        }
    )

    # After refine, always answer
    qa_builder.add_edge("qa_refine", "qa_answer")
    qa_builder.add_edge("qa_answer", END)

    return qa_builder.compile()


# Export the compiled QA graph
qa_graph = build_qa_graph()
