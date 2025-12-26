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
from app.agent.utils import setup_langsmith
import re
import string
from typing import List, Dict, Any
from collections import Counter

setup_langsmith()
dataset_name = "qasper-qa-e2e"


# ============================================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================================

def normalize_answer(text: str) -> str:
    """
    Normalize text for comparison.
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    """
    text = str(text)
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return normalize_answer(text).split()


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        F1 score between 0.0 and 1.0
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    # Count token occurrences
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    
    # Calculate common tokens
    common_tokens = pred_counter & gt_counter
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0.0
    
    # Calculate precision and recall
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    
    # Calculate F1
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Binary score: 1.0 if normalized strings match exactly, 0.0 otherwise.
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_evidence_recall(retrieved_segments: List[str], 
                            ground_truth_evidence: List[str]) -> float:
    """
    Calculate what fraction of ground truth evidence spans appear in retrieved segments.
    """
    if not ground_truth_evidence or len(ground_truth_evidence) == 0:
        return 0.0
    
    if not retrieved_segments or len(retrieved_segments) == 0:
        return 0.0
    
    # Normalize all segments for comparison
    normalized_retrieved = [normalize_answer(seg) for seg in retrieved_segments]
    normalized_evidence = [normalize_answer(ev) for ev in ground_truth_evidence]
    
    # Count how many evidence spans are found
    found_count = 0
    valid_evidence_count = 0  # Track how many evidence items were actually valid
    
    for evidence in normalized_evidence:
        evidence_tokens = set(evidence.split())
        
        # FIX: Skip empty evidence to prevent division by zero
        if len(evidence_tokens) == 0:
            continue
            
        valid_evidence_count += 1
        
        for retrieved in normalized_retrieved:
            retrieved_tokens = set(retrieved.split())
            
            # If at least 50% of evidence tokens are in retrieved segment
            overlap = len(evidence_tokens & retrieved_tokens)
            if overlap / len(evidence_tokens) >= 0.5:
                found_count += 1
                break
    
    # Avoid division by zero if all evidence items were empty
    if valid_evidence_count == 0:
        return 0.0
        
    # You can choose to divide by the original count or the valid count.
    # Usually, dividing by original count is stricter (penalizes empty/bad data).
    # If you want to be lenient, use valid_evidence_count.
    return found_count / len(ground_truth_evidence)


# ============================================================================
# ANSWER TYPE-SPECIFIC EVALUATORS
# ============================================================================

def evaluate_unanswerable(prediction: str, ground_truth: dict) -> dict:
    """
    Check if model correctly identifies unanswerable questions.
    
    Args:
        prediction: Model's answer
        ground_truth: Ground truth data with answer_type
        
    Returns:
        Dictionary with unanswerable_accuracy metric
    """
    # Keywords that indicate unanswerable
    unanswerable_keywords = [
        'cannot', 'unanswerable', 'insufficient', 'not found',
        'no information', 'not mentioned', 'unclear', 'unable to answer'
    ]
    
    pred_lower = prediction.lower()
    is_unanswerable = any(keyword in pred_lower for keyword in unanswerable_keywords)
    
    # Score 1.0 if model correctly identifies as unanswerable
    score = 1.0 if is_unanswerable else 0.0
    
    return {
        "key": "unanswerable_accuracy",
        "score": score
    }


def evaluate_yes_no(prediction: str, ground_truth: dict) -> dict:
    """
    Check if model correctly answers yes/no questions.
    
    Args:
        prediction: Model's answer
        ground_truth: Ground truth data with ground_truth_answer
        
    Returns:
        Dictionary with yes_no_accuracy metric
    """
    pred_lower = prediction.lower()
    gt_answer = ground_truth.get("ground_truth_answer", "").lower()
    
    # Extract yes/no from prediction
    has_yes = 'yes' in pred_lower
    has_no = 'no' in pred_lower and 'not' not in pred_lower[:pred_lower.find('no')] if 'no' in pred_lower else False
    
    # Determine predicted answer
    if has_yes and not has_no:
        pred_answer = "yes"
    elif has_no and not has_yes:
        pred_answer = "no"
    else:
        # Ambiguous, check which appears first
        yes_pos = pred_lower.find('yes') if has_yes else float('inf')
        no_pos = pred_lower.find('no') if has_no else float('inf')
        pred_answer = "yes" if yes_pos < no_pos else "no" if no_pos < float('inf') else ""
    
    # Score 1.0 if match
    score = 1.0 if pred_answer == gt_answer else 0.0
    
    return {
        "key": "yes_no_accuracy",
        "score": score
    }


def evaluate_answer(prediction: str, ground_truth: dict) -> dict:
    """
    Evaluate extractive or free-form answers.
    
    Args:
        prediction: Model's answer
        ground_truth: Ground truth data with ground_truth_answer
        
    Returns:
        Dictionary with f1_score and exact_match metrics
    """
    gt_answer = ground_truth.get("ground_truth_answer", "")
    # Handle non-string ground truth answers
    if not isinstance(gt_answer, str):
        gt_answer = str(gt_answer) if gt_answer is not None else ""
    
    f1 = compute_f1(prediction, gt_answer)
    em = compute_exact_match(prediction, gt_answer)
      
    return {
        "key": "answer_quality",
        "score": f1,  # Primary score for LangSmith
        "f1_score": f1,
        "exact_match": em
    }


# ============================================================================
# MAIN EVALUATOR
# ============================================================================

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
