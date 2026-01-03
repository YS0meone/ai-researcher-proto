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
from app.core.config import settings
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langsmith import Client


setup_langsmith()
dataset_name = "qasper-qa-e2e"

eval_model = init_chat_model(model=settings.EVAL_MODEL_NAME, api_key=settings.GEMINI_API_KEY)


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
    # Note: retrieved_segments is now a List[str] (raw tool outputs), not List[Dict]
    retrieved_segments = result_state.get("retrieved_segments", [])
    
    return {
        "answer": answer,
        "metadata": {
            "retrieved_segments": retrieved_segments,
            "num_segments": len(retrieved_segments),
            "selected_ids": result_state.get("selected_ids", []),
            "retrieval_queries": result_state.get("retrieval_queries", [])
        }
    }

def qa_e2e_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    prompt = f"""
    You are a QA evaluator which specializes in evaluating the quality of the answer to a question related to a specific scientific paper.
    You are given a question, a ground truth answer, a model's answer, retrieved segments, and ground truth evidence.
    You need to evaluate the quality of the model's answer and return the score.
    The score is a number between 0 and 1, where 1 is the best score.
    The score is calculated based on the following criteria:
    - The model's answer is complete and covers all the aspects of the ground truth answer.
    - The model's answer is grounded in the retrieved segments and does not hallucinate.
    - Even though the model's answer is not like the ground truth answer, as long as it covers all the aspects of the ground truth answer, and not hallucinating, it should be given a score of 1.
    
    You need to return both the score and the reasoning for the score.
    For reasoning, you need to determine the catergory of the error (if any). The categories are:
    - Retrieval Error: The retrieved segments does not contain the ground truth evidence.
    - Hallucination: The model's answer is not grounded in the retrieved segments and hallucinates.
    - Incomplete: The model's answer is not complete and does not cover all the aspects of the ground truth answer.
    - No Error: The model's answer is complete, grounded in the retrieved segments, and does not hallucinate.
    - Other: The model's answer is not related to the question or the ground truth answer.
    In the reasoning mention how the retrieved segements is affecting the score.

    Question: {inputs["question"]}
    Ground truth answer: {reference_outputs["ground_truth_answer"]}
    Model's answer: {outputs["answer"]}
    Retrieved segments: {outputs["metadata"]["retrieved_segments"]}
    Ground truth evidence: {reference_outputs["ground_truth_evidence"]}
    """

    class QaEvaluatorResponse(BaseModel):
        score: float = Field(ge=0.0, le=1.0)
        reasoning: str

    structured_eval_model = eval_model.with_structured_output(QaEvaluatorResponse)
    response = structured_eval_model.invoke(prompt)
    return {
        "score": response.score,
        "comment": response.reasoning  # LangSmith UI displays 'comment' field
    }


def retrieval_evaluator(outputs: dict, reference_outputs: dict) -> float:
    retrieved_segments = outputs["metadata"]["retrieved_segments"]
    ground_truth_evidence = reference_outputs["ground_truth_evidence"]
    hit = 0
    for gt in ground_truth_evidence:
        if gt in retrieved_segments:
            hit += 1
    return hit / len(ground_truth_evidence) if len(ground_truth_evidence) > 0 else 0.0

def main():
    """Run QASPER evaluation."""
    print("Starting QASPER evaluation...")
    print(f"Dataset: {dataset_name}")
    client = Client()
    dataset = client.read_dataset(dataset_name=dataset_name)
    results = evaluate(
        qa_agent_wrapper,
        data=client.list_examples(dataset_id=dataset.id),
        evaluators=[qa_e2e_evaluator, retrieval_evaluator],
        max_concurrency=10,
    )


if __name__ == "__main__":
    main()
