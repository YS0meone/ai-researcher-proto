"""
BEIR SCIDOCS Matched Subset Evaluation
Uses only the 1,649 papers that exist in both BEIR and our arXiv dataset
"""
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

from app.tools.search import (
    hybrid_search_papers_impl,
    semantic_search_papers_impl,
    keyword_search_papers_impl
)


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single query"""
    query: str
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    recall_at_20: float
    ndcg_at_10: float
    mrr: float
    latency_ms: float
    num_results: int


class MatchedBEIREvaluation:
    """Evaluate using matched BEIR subset (papers in both BEIR and our arXiv dataset)"""
    
    def __init__(self):
        """Initialize with matched subset"""
        self.eval_dir = Path(__file__).parent
        self.queries = {}
        self.corpus = {}
        self.qrels = {}
        self.corpus_to_arxiv = {}
        self.arxiv_to_corpus = {}
        
        self.load_matched_dataset()
    
    def load_matched_dataset(self):
        """Load the matched subset dataset"""
        print("\nLoading matched BEIR subset...")
        
        # Load ID mapping
        mapping_file = self.eval_dir / "data" / "scidocs" / "arxiv_id_mapping.json"
        with open(mapping_file, 'r') as f:
            self.corpus_to_arxiv = json.load(f)
            self.arxiv_to_corpus = {v: k for k, v in self.corpus_to_arxiv.items()}
        
        print(f"Loaded {len(self.corpus_to_arxiv):,} ID mappings")
        
        # Load matched corpus
        corpus_file = self.eval_dir / "data" / "scidocs" / "corpus_matched.jsonl"
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                self.corpus[doc['_id']] = doc
        
        print(f"Loaded {len(self.corpus):,} corpus documents")
        
        # Load matched queries
        queries_file = self.eval_dir / "data" / "scidocs" / "queries_matched.jsonl"
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line)
                self.queries[query['_id']] = query['text']
        
        print(f"Loaded {len(self.queries):,} queries")
        
        # Load matched qrels
        qrels_file = self.eval_dir / "data" / "scidocs" / "qrels" / "test_matched.tsv"
        with open(qrels_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    query_id = parts[0]
                    corpus_id = parts[1]
                    score = int(parts[2])
                    
                    if query_id not in self.qrels:
                        self.qrels[query_id] = {}
                    self.qrels[query_id][corpus_id] = score
        
        print(f"Loaded relevance judgments for {len(self.qrels):,} queries\n")
    
    def evaluate_search_method(self, search_func, method_name: str, 
                              max_queries: int = None) -> Dict[str, Any]:
        """
        Evaluate a search method
        
        Args:
            search_func: Search function (query: str, limit: int) -> List[Dict]
            method_name: Name of search method
            max_queries: Maximum number of queries to evaluate (None = all)
        
        Returns:
            Dictionary with aggregated metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}\n")
        
        all_metrics = []
        queries_to_eval = list(self.queries.items())
        
        if max_queries:
            queries_to_eval = queries_to_eval[:max_queries]
        
        for query_id, query_text in tqdm(queries_to_eval, desc=f"{method_name}"):
            # Get ground truth
            if query_id not in self.qrels:
                continue
            
            relevant_corpus_ids = [cid for cid, score in self.qrels[query_id].items() if score > 0]
            if not relevant_corpus_ids:
                continue
            
            # Map corpus IDs to arXiv IDs
            relevant_arxiv_ids = [
                self.corpus_to_arxiv[cid] 
                for cid in relevant_corpus_ids 
                if cid in self.corpus_to_arxiv
            ]
            
            if not relevant_arxiv_ids:
                continue
            
            # Execute search
            start_time = time.time()
            try:
                results = search_func(query_text, limit=20)
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract arXiv IDs from results
                # Try both 'id' and 'arxiv_id' fields (ES returns 'arxiv_id')
                retrieved_arxiv_ids = [
                    r.get('id') or r.get('arxiv_id') or r.get('paper_id', '')
                    for r in results 
                    if r.get('id') or r.get('arxiv_id') or r.get('paper_id')
                ]
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    query_text, retrieved_arxiv_ids, relevant_arxiv_ids, latency_ms
                )
                all_metrics.append(metrics)
                
            except KeyboardInterrupt:
                print(f"\n\nEvaluation interrupted by user")
                break
            except Exception as e:
                # Continue on error but log it
                continue
        
        # Aggregate metrics
        return self._aggregate_metrics(all_metrics, method_name)
    
    def _calculate_metrics(self, query: str, retrieved_ids: List[str], 
                          relevant_ids: List[str], latency_ms: float) -> EvalMetrics:
        """Calculate metrics for a single query"""
        relevant_set = set(relevant_ids)
        
        # Precision@K
        def precision_at_k(k):
            if not retrieved_ids or k == 0:
                return 0.0
            retrieved_k = retrieved_ids[:k]
            return sum(1 for rid in retrieved_k if rid in relevant_set) / k
        
        # Recall@K
        def recall_at_k(k):
            if not relevant_ids:
                return 0.0
            retrieved_k = retrieved_ids[:k]
            return sum(1 for rid in retrieved_k if rid in relevant_set) / len(relevant_ids)
        
        # NDCG@K
        def ndcg_at_k(k):
            if not relevant_ids:
                return 0.0
            retrieved_k = retrieved_ids[:k]
            
            # DCG
            dcg = 0.0
            for i, rid in enumerate(retrieved_k):
                if rid in relevant_set:
                    dcg += 1.0 / np.log2(i + 2)
            
            # IDCG (ideal)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
            
            return dcg / idcg if idcg > 0 else 0.0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        
        return EvalMetrics(
            query=query,
            precision_at_5=precision_at_k(5),
            precision_at_10=precision_at_k(10),
            recall_at_10=recall_at_k(10),
            recall_at_20=recall_at_k(20),
            ndcg_at_10=ndcg_at_k(10),
            mrr=mrr,
            latency_ms=latency_ms,
            num_results=len(retrieved_ids)
        )
    
    def _aggregate_metrics(self, all_metrics: List[EvalMetrics], 
                          method_name: str) -> Dict[str, Any]:
        """Aggregate metrics across all queries"""
        if not all_metrics:
            return {"error": "No valid queries evaluated"}
        
        # Calculate averages
        avg_metrics = {
            "method": method_name,
            "num_queries": len(all_metrics),
            "precision@5": np.mean([m.precision_at_5 for m in all_metrics]),
            "precision@10": np.mean([m.precision_at_10 for m in all_metrics]),
            "recall@10": np.mean([m.recall_at_10 for m in all_metrics]),
            "recall@20": np.mean([m.recall_at_20 for m in all_metrics]),
            "ndcg@10": np.mean([m.ndcg_at_10 for m in all_metrics]),
            "mrr": np.mean([m.mrr for m in all_metrics]),
            "avg_latency_ms": np.mean([m.latency_ms for m in all_metrics]),
            "median_latency_ms": np.median([m.latency_ms for m in all_metrics]),
            "p95_latency_ms": np.percentile([m.latency_ms for m in all_metrics], 95),
            "p99_latency_ms": np.percentile([m.latency_ms for m in all_metrics], 99),
        }
        
        # Print results
        print(f"\nResults for {method_name}:")
        print(f"   Queries evaluated: {avg_metrics['num_queries']}")
        print(f"   Precision@5:  {avg_metrics['precision@5']:.2%}")
        print(f"   Precision@10: {avg_metrics['precision@10']:.2%}")
        print(f"   Recall@10:    {avg_metrics['recall@10']:.2%}")
        print(f"   Recall@20:    {avg_metrics['recall@20']:.2%}")
        print(f"   NDCG@10:      {avg_metrics['ndcg@10']:.2%}")
        print(f"   MRR:          {avg_metrics['mrr']:.3f}")
        print(f"   Latency:      {avg_metrics['avg_latency_ms']:.0f}ms (avg), "
              f"{avg_metrics['median_latency_ms']:.0f}ms (median)")
        
        return avg_metrics


def main():
    """Run evaluation on matched subset"""
    print("\n" + "="*60)
    print(" BEIR SCIDOCS Matched Subset Evaluation")
    print(" (Only papers in both BEIR and our arXiv dataset)")
    print("="*60)
    
    # Initialize evaluator
    evaluator = MatchedBEIREvaluation()
    
    # Define search methods
    search_methods = {
        "Hybrid Search": lambda q, limit: hybrid_search_papers_impl(q, limit=limit),
        "Semantic Search": lambda q, limit: semantic_search_papers_impl(q, limit=limit),
        "Keyword Search": lambda q, limit: keyword_search_papers_impl(q, limit=limit),
    }
    
    # Run evaluation
    all_results = {}
    for method_name, search_func in search_methods.items():
        try:
            results = evaluator.evaluate_search_method(search_func, method_name)
            all_results[method_name] = results
        except Exception as e:
            print(f"\nError evaluating {method_name}: {e}")
            continue
    
    # Save results
    output_file = Path(__file__).parent / "results" / "beir_matched_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print(" Summary Comparison")
    print("="*60)
    print(f"{'Method':<20} {'NDCG@10':>10} {'Precision@10':>12} {'Latency (ms)':>12}")
    print("-" * 60)
    
    for method_name, results in all_results.items():
        if "error" not in results:
            print(f"{method_name:<20} "
                  f"{results['ndcg@10']:>9.2%} "
                  f"{results['precision@10']:>11.2%} "
                  f"{results['avg_latency_ms']:>11.0f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
