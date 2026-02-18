"""
QASPER Dataset Evaluation Script

Evaluates the QA agent on the QASPER benchmark with proper metrics:
- F1 Score (token-level overlap)
- Exact Match (normalized string matching)
- Evidence Recall (retrieval quality)
- Answer Type Accuracy (unanswerable, yes/no, extractive, free-form)
"""

from langsmith import evaluate
from app.agent.qa import qa_graph
from langchain.messages import HumanMessage
import re
import string
from typing import List
from collections import Counter
from app.core.config import settings
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langsmith import Client
from app.agent.qa_baseline import qa_baseline
from app.agent.prompts import LLM_AS_JUDGE_PROMPT
from app.core.schema import S2Paper

dataset_name = "qasper-qa-e2e"

eval_model = init_chat_model(model=settings.EVAL_MODEL_NAME, api_key=settings.GEMINI_API_KEY)


def qa_agent_wrapper(dataset_input: dict) -> dict:
    """
    Enhanced wrapper that returns answer and retrieval metadata.
    
    Args:
        dataset_input: Dictionary with 'paper_id' and 'question'
        
    Returns:
        Dictionary with 'answer' and 'metadata'
    """
    papers = [S2Paper(paperId=dataset_input["paper_id"], abstract=dataset_input["abstract"])]

    initial_state = {
        "messages": [HumanMessage(content=dataset_input["question"])],
        "user_query": dataset_input["question"],
        "selected_paper_ids": [dataset_input["paper_id"]],
        "papers": papers
    }
    
    result_state = qa_graph.invoke(initial_state)
    
    # Extract answer
    answer = result_state.get("final_answer", "something wrong with the qa graph while executing...")
        
    # Extract retrieved segments for evidence recall
    # Note: retrieved_segments is now a List[str] (raw tool outputs), not List[Dict]
    retrieved_segments = [document.page_content for document in result_state.get("evidences", [])]
    
    return {
        "answer": answer,
        "metadata": {
            "retrieved_segments": retrieved_segments,
            "num_segments": len(retrieved_segments),
            "selected_ids": result_state.get("selected_ids", []),
            "retrieval_queries": result_state.get("retrieval_queries", [])
        }
    }

def qa_baseline_wrapper(dataset_input: dict) -> dict:
    """
    Enhanced wrapper that returns answer and retrieval metadata.
    """
    initial_state = {
        "messages": [HumanMessage(content=dataset_input["question"])],
        "selected_ids": [dataset_input["paper_id"]]
    }
    
    result_state = qa_baseline.invoke(initial_state)
    
    return {
        "answer": result_state["messages"][-1].content,
        "metadata": {
            "retrieved_segments": [],
            "num_segments": 0,
            "selected_ids": [],
            "retrieval_queries": [],
            "reasoning": result_state["reasoning"],
        }
    }


def qa_e2e_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    prompt = LLM_AS_JUDGE_PROMPT.format(
        question=inputs["question"],
        ground_truth_answer=reference_outputs.get("ground_truth_answer", ""),
        ground_truth_evidence=reference_outputs.get("ground_truth_evidence", []),
        retrieved_evidence=outputs.get("metadata", {}).get("retrieved_segments", []),
        generated_answer=outputs.get("answer", ""),
    )

    class QaEvaluatorResponse(BaseModel):
        accuracy_score: int = Field(ge=1, le=5)
        synthesis_score: int = Field(ge=1, le=5)
        comprehensiveness_score: int = Field(ge=1, le=5)
        overall_score: float = Field(ge=1.0, le=5.0)

    structured_eval_model = eval_model.with_structured_output(QaEvaluatorResponse)
    response = structured_eval_model.invoke(prompt)

    # Normalize overall_score from 1-5 to 0-1 for LangSmith
    normalized_score = (response.overall_score - 1) / 4

    return {
        "key": "llm_judge",
        "score": normalized_score,
        "comment": (
            f"accuracy={response.accuracy_score}/5, "
            f"synthesis={response.synthesis_score}/5, "
            f"comprehensiveness={response.comprehensiveness_score}/5, "
            f"overall={response.overall_score}/5"
        )
    }

def retrieval_evaluator(outputs: dict, reference_outputs: dict) -> float:
    retrieved_segments = outputs.get("metadata", {}).get("retrieved_segments", [])
    ground_truth_evidence = reference_outputs["ground_truth_evidence"]
    hit = 0
    for gt in ground_truth_evidence:
        if gt in retrieved_segments:
            hit += 1
    return hit / len(ground_truth_evidence) if len(ground_truth_evidence) > 0 else 0.0

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run QASPER evaluation")
    parser.add_argument(
        "--agent", choices=["qa", "baseline"], default="qa",
        help="Which agent to evaluate (default: qa)"
    )
    parser.add_argument(
        "--split", default=None,
        help="Dataset split to evaluate on (e.g. 'test'). Omit to run on all examples."
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=1,
        help="Max concurrent evaluations (default: 1)"
    )
    args = parser.parse_args()

    wrapper = qa_agent_wrapper if args.agent == "qa" else qa_baseline_wrapper

    list_kwargs = {"dataset_name": dataset_name}
    if args.split:
        list_kwargs["splits"] = [args.split]

    experiment_prefix = f"{args.agent}-{'all' if not args.split else args.split}"

    print(f"Agent:   {args.agent}")
    print(f"Split:   {args.split or 'all'}")

    client = Client()
    evaluate(
        wrapper,
        data=client.list_examples(**list_kwargs),
        evaluators=[qa_e2e_evaluator, retrieval_evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=args.max_concurrency,
    )


if __name__ == "__main__":
    main()
