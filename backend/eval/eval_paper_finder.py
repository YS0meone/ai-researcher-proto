from langsmith import evaluate
from app.agent.paper_finder import pf_graph
from langchain.messages import HumanMessage
from app.agent.utils import setup_langsmith
import re
import string
from typing import List, Dict, Any
from collections import Counter

setup_langsmith()
dataset_name = "qasper-qa-e2e"

def qasper_evaluator(run, example) -> dict:
    """
    Main evaluator that handles all answer types.
    
    Args:
        run: LangSmith run with prediction and metadata
        example: Ground truth from dataset
        
    Returns:
        Dictionary with all relevant metrics
    """
    # Extract prediction
    prediction = run.outputs.get("answer", "") if isinstance(run.outputs, dict) else str(run.outputs)
    
    # Extract ground truth
    ground_truth = example.outputs
    answer_type = ground_truth.get("answer_type", "free_form_answer")
    
    results = {"answer_type": answer_type}
    
    # Route to appropriate evaluator based on answer type
    if answer_type == "unanswerable":
        results.update(evaluate_unanswerable(prediction, ground_truth))
    elif answer_type == "yes_no":
        results.update(evaluate_yes_no(prediction, ground_truth))
    else:  # extractive_spans or free_form_answer
        results.update(evaluate_answer(prediction, ground_truth))
    
    # Add evidence recall if available
    if "ground_truth_evidence" in ground_truth and isinstance(run.outputs, dict):
        retrieved_segments = run.outputs.get("metadata", {}).get("retrieved_segments", [])
        if retrieved_segments and len(ground_truth["ground_truth_evidence"]) > 0:
            evidence_recall = compute_evidence_recall(
                retrieved_segments,
                ground_truth["ground_truth_evidence"]
            )
            results["evidence_recall"] = evidence_recall
    
    return results


# ============================================================================
# AGENT WRAPPER
# ============================================================================

def qa_agent_wrapper(dataset_input: dict) -> dict:
    """
    Enhanced wrapper that returns answer and retrieval metadata.
    
    Args:
        dataset_input: Dictionary with 'paper_id' and 'question'
        
    Returns:
        Dictionary with 'answer' and 'metadata'
    """
    initial_state = {
        "messages": [HumanMessage(content=dataset_input["question"])],
        "selected_ids": [dataset_input["paper_id"]]
    }
    
    result_state = qa_graph.invoke(initial_state)
    
    # Extract answer
    answer = result_state["messages"][-1].content if result_state.get("messages") else "No answer generated"
    
    # Extract retrieved segments for evidence recall
    retrieved_segments = [
        seg.get("supporting_detail", "") 
        for seg in result_state.get("retrieved_segments", [])
    ]
    
    return {
        "answer": answer,
        "metadata": {
            "retrieved_segments": retrieved_segments,
            "num_segments": len(retrieved_segments)
        }
    }


# ============================================================================
# RESULT ANALYSIS
# ============================================================================

def analyze_results(eval_results):
    """
    Aggregate evaluation results by answer type and overall.
    
    Args:
        eval_results: LangSmith evaluation results (iterable of dicts)
    """
    print("\n" + "="*60)
    print("QASPER EVALUATION RESULTS")
    print("="*60)
    
    # Aggregate metrics
    metrics = {
        "f1_scores": [],
        "exact_matches": [],
        "evidence_recalls": [],
        "unanswerable_accuracies": [],
        "yes_no_accuracies": [],
    }
    
    answer_type_counts = {}
    
    # Collect results
    for result in eval_results:
        # 1. Get Answer Type directly from the Ground Truth (Example)
        # 'result' is a dict containing {'run': ..., 'example': ..., 'evaluation_results': ...}
        example = result.get("example")
        ground_truth = example.outputs if hasattr(example, "outputs") else {}
        answer_type = ground_truth.get("answer_type", "unknown")
        
        # Track counts
        answer_type_counts[answer_type] = answer_type_counts.get(answer_type, 0) + 1
        
        # 2. Process Feedback (EvaluationResult objects)
        feedback = result.get("evaluation_results", {}).get("results", [])
        
        for fb in feedback:
            # FIX: Access attributes with dot notation, not .get()
            key = fb.key 
            score = fb.score
            
            # Access extra fields from evaluator_info if available (LangSmith stores extra dict keys here)
            # Some versions might store flat fields in evaluator_info
            extra_info = getattr(fb, "evaluator_info", {}) or {}
            
            # --- Collect Metrics ---
            
            # F1 Score (Primary score for answer_quality)
            if key == "answer_quality" and score is not None:
                metrics["f1_scores"].append(score)
                
            # Exact Match (often stored in extra info)
            if "exact_match" in extra_info:
                metrics["exact_matches"].append(extra_info["exact_match"])
            elif key == "exact_match": # If you returned it as a separate key
                metrics["exact_matches"].append(score)

            # Evidence Recall
            if "evidence_recall" in extra_info:
                metrics["evidence_recalls"].append(extra_info["evidence_recall"])
            elif key == "evidence_recall":
                 metrics["evidence_recalls"].append(score)
                
            # Unanswerable Accuracy
            if key == "unanswerable_accuracy" and score is not None:
                metrics["unanswerable_accuracies"].append(score)
                
            # Yes/No Accuracy
            if key == "yes_no_accuracy" and score is not None:
                metrics["yes_no_accuracies"].append(score)

    # Print overall metrics
    print("\nOVERALL METRICS:")
    print("-" * 60)
    
    if metrics["f1_scores"]:
        avg_f1 = sum(metrics["f1_scores"]) / len(metrics["f1_scores"])
        print(f"Average F1 Score:        {avg_f1:.3f}")
    
    if metrics["exact_matches"]:
        avg_em = sum(metrics["exact_matches"]) / len(metrics["exact_matches"])
        print(f"Average Exact Match:     {avg_em:.3f}")
    
    if metrics["evidence_recalls"]:
        avg_recall = sum(metrics["evidence_recalls"]) / len(metrics["evidence_recalls"])
        print(f"Average Evidence Recall: {avg_recall:.3f}")
    
    if metrics["unanswerable_accuracies"]:
        avg_unans = sum(metrics["unanswerable_accuracies"]) / len(metrics["unanswerable_accuracies"])
        print(f"Unanswerable Accuracy:   {avg_unans:.3f}")
    
    if metrics["yes_no_accuracies"]:
        avg_yn = sum(metrics["yes_no_accuracies"]) / len(metrics["yes_no_accuracies"])
        print(f"Yes/No Accuracy:         {avg_yn:.3f}")
    
    # Print answer type breakdown
    print("\nANSWER TYPE DISTRIBUTION:")
    print("-" * 60)
    for atype, count in sorted(answer_type_counts.items()):
        print(f"{atype:20s}: {count:4d}")
    
    print("\n" + "="*60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run QASPER evaluation."""
    print("Starting QASPER evaluation...")
    print(f"Dataset: {dataset_name}")
    
    results = evaluate(
        qa_agent_wrapper,
        data=dataset_name,
        evaluators=[qasper_evaluator],
        max_concurrency=1,  # Avoid rate limits and ensure reproducibility
    )
    
    # Analyze and display results
    analyze_results(results)
    
    # Save detailed results
    output_file = "qasper_eval_results.csv"
    try:
        results.to_csv(output_file)
        print(f"\n✅ Evaluation complete! Detailed results saved to {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results to CSV: {e}")
        print("Results object:", results)


if __name__ == "__main__":
    main()
