"""
BEIR SCIDOCS Evaluation for Paper Finder
Uses official BEIR benchmark dataset for academic paper retrieval
Dataset: SCIDOCS (1,000 queries with human-annotated ground truth)
"""
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import json
import time
import os
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

from app.tools.search import (
    hybrid_search_papers_impl,
    semantic_search_papers_impl,
    keyword_search_papers_impl
)


@dataclass
class SearchQuery:
    """Represents a test search query"""
    query: str
    relevant_paper_ids: List[str]  # Ground truth relevant papers
    category: str  # Query category (e.g., "method", "dataset", "application")
    difficulty: str  # "easy", "medium", "hard"


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single query"""
    query: str
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    recall_at_20: float
    ndcg_at_10: float  # Normalized Discounted Cumulative Gain
    mrr: float  # Mean Reciprocal Rank
    latency_ms: float
    num_results: int


class BEIRBenchmark:
    """BEIR SCIDOCS benchmark for academic paper retrieval"""
    
    def __init__(self, dataset_name: str = "scidocs", use_beir: bool = True):
        """
        Initialize benchmark
        
        Args:
            dataset_name: BEIR dataset name (default: scidocs)
            use_beir: Whether to use official BEIR data
        """
        self.queries: List[SearchQuery] = []
        self.dataset_name = dataset_name
        self.use_beir = use_beir
        self.beir_corpus: Dict[str, Dict] = {}
        self.arxiv_to_beir_mapping: Dict[str, str] = {}
        
        if use_beir:
            self.load_beir_dataset()
    
    def load_beir_dataset(self):
        """Load official BEIR SCIDOCS dataset"""
        try:
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader
            
            print(f"\n📥 Downloading BEIR {self.dataset_name} dataset...")
            dataset_path = util.download_and_unzip(
                f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip",
                "eval/data"
            )
            
            # Load corpus, queries, and qrels
            corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split="test")
            self.beir_corpus = corpus
            
            print(f"✅ Loaded {len(queries)} queries with ground truth")
            print(f"   Corpus: {len(corpus)} documents")
            
            # Build title-based mapping for ID matching
            print("\n🔗 Building title-based ID mapping...")
            self._build_title_mapping()
            
            # Convert to SearchQuery format
            for query_id, query_text in queries.items():
                if query_id in qrels:
                    # Get relevant paper IDs (score >= 1)
                    beir_relevant_ids = [doc_id for doc_id, score in qrels[query_id].items() if score >= 1]
                    
                    # Map BEIR IDs to arXiv IDs
                    relevant_ids = []
                    for beir_id in beir_relevant_ids:
                        if beir_id in self.arxiv_to_beir_mapping.values():
                            # Find arXiv ID for this BEIR ID
                            for arxiv_id, mapped_beir_id in self.arxiv_to_beir_mapping.items():
                                if mapped_beir_id == beir_id:
                                    relevant_ids.append(arxiv_id)
                                    break
                    
                    # If no mapping found, use BEIR IDs directly (for relative comparison)
                    if not relevant_ids:
                        relevant_ids = beir_relevant_ids
                    
                    if relevant_ids:
                        self.queries.append(SearchQuery(
                            query=query_text,
                            relevant_paper_ids=relevant_ids,
                            category="academic",
                            difficulty="medium"
                        ))
            
            print(f"✅ Prepared {len(self.queries)} evaluation queries")
            print(f"   Mapped {len(self.arxiv_to_beir_mapping)} arXiv IDs\n")
            
        except ImportError:
            print("\n⚠️  BEIR library not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "beir"])
            print("✅ BEIR installed. Please run again.\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error loading BEIR: {e}")
            print("   Falling back to manual queries...\n")
            self.use_beir = False
    
    def _build_title_mapping(self):
        """Build mapping between arXiv IDs and BEIR IDs using titles"""
        try:
            from app.services.elasticsearch import ElasticsearchService
            from app.core.config import settings
            
            es_service = ElasticsearchService(settings.elasticsearch_config)
            
            # Get sample papers from ES to build mapping
            for beir_id, beir_doc in list(self.beir_corpus.items())[:1000]:  # Sample for speed
                beir_title = beir_doc.get('title', '').lower().strip()
                if not beir_title:
                    continue
                
                # Search ES by title
                results = es_service.text_search(beir_title, limit=1)
                if results and len(results) > 0:
                    arxiv_id = results[0].get('id', '')
                    if arxiv_id:
                        self.arxiv_to_beir_mapping[arxiv_id] = beir_id
        except Exception as e:
            print(f"   ⚠️  Title mapping failed: {e}")
            print(f"   ℹ️  Will use BEIR IDs directly for relative evaluation")
    
    def load_queries_from_file(self, file_path: str):
        """Load test queries from JSON file (fallback)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data['queries']:
                self.queries.append(SearchQuery(**item))
        print(f"Loaded {len(self.queries)} test queries")
    
    def add_query(self, query: str, relevant_ids: List[str], 
                  category: str = "general", difficulty: str = "medium"):
        """Manually add a test query"""
        self.queries.append(SearchQuery(
            query=query,
            relevant_paper_ids=relevant_ids,
            category=category,
            difficulty=difficulty
        ))
    
    def precision_at_k(self, retrieved_ids: List[str], 
                       relevant_ids: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if not retrieved_ids or k == 0:
            return 0.0
        
        retrieved_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
        return relevant_retrieved / k
    
    def recall_at_k(self, retrieved_ids: List[str], 
                    relevant_ids: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant_ids:
            return 0.0
        
        retrieved_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
        return relevant_retrieved / len(relevant_ids)
    
    def ndcg_at_k(self, retrieved_ids: List[str], 
                  relevant_ids: List[str], k: int) -> float:
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)"""
        if not retrieved_ids or not relevant_ids:
            return 0.0
        
        retrieved_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        
        # DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k):
            if doc_id in relevant_set:
                # Binary relevance: 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1
        
        # IDCG (Ideal DCG) - all relevant docs at top
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def mean_reciprocal_rank(self, retrieved_ids: List[str], 
                             relevant_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        relevant_set = set(relevant_ids)
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def evaluate_query(self, query: SearchQuery, 
                       search_function, **search_kwargs) -> EvalMetrics:
        """Evaluate a single query"""
        # Measure latency
        start_time = time.time()
        results = search_function(query.query, **search_kwargs)
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract paper IDs from results
        retrieved_ids = []
        for result in results:
            if 'arxiv_id' in result:
                retrieved_ids.append(result['arxiv_id'])
            elif 'error' in result:
                print(f"Search error: {result['error']}")
                break
        
        # Calculate metrics
        p_at_5 = self.precision_at_k(retrieved_ids, query.relevant_paper_ids, 5)
        p_at_10 = self.precision_at_k(retrieved_ids, query.relevant_paper_ids, 10)
        r_at_10 = self.recall_at_k(retrieved_ids, query.relevant_paper_ids, 10)
        r_at_20 = self.recall_at_k(retrieved_ids, query.relevant_paper_ids, 20)
        ndcg_10 = self.ndcg_at_k(retrieved_ids, query.relevant_paper_ids, 10)
        mrr = self.mean_reciprocal_rank(retrieved_ids, query.relevant_paper_ids)
        
        return EvalMetrics(
            query=query.query,
            precision_at_5=p_at_5,
            precision_at_10=p_at_10,
            recall_at_10=r_at_10,
            recall_at_20=r_at_20,
            ndcg_at_10=ndcg_10,
            mrr=mrr,
            latency_ms=latency_ms,
            num_results=len(retrieved_ids)
        )
    
    def run_evaluation(self, search_function, 
                       search_name: str = "Unknown",
                       **search_kwargs) -> Dict[str, Any]:
        """
        Run full evaluation on all queries
        
        Args:
            search_function: Function to evaluate (e.g., hybrid_search_papers_impl)
            search_name: Name of the search method for reporting
            **search_kwargs: Additional arguments passed to search function
        
        Returns:
            Dictionary with aggregate metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {search_name}")
        print(f"{'='*60}")
        
        all_metrics: List[EvalMetrics] = []
        
        for query in tqdm(self.queries, desc=f"Testing {search_name}"):
            metrics = self.evaluate_query(query, search_function, **search_kwargs)
            all_metrics.append(metrics)
        
        # Aggregate results
        aggregate = {
            'search_method': search_name,
            'num_queries': len(all_metrics),
            'avg_precision_at_5': np.mean([m.precision_at_5 for m in all_metrics]),
            'avg_precision_at_10': np.mean([m.precision_at_10 for m in all_metrics]),
            'avg_recall_at_10': np.mean([m.recall_at_10 for m in all_metrics]),
            'avg_recall_at_20': np.mean([m.recall_at_20 for m in all_metrics]),
            'avg_ndcg_at_10': np.mean([m.ndcg_at_10 for m in all_metrics]),
            'avg_mrr': np.mean([m.mrr for m in all_metrics]),
            'avg_latency_ms': np.mean([m.latency_ms for m in all_metrics]),
            'median_latency_ms': np.median([m.latency_ms for m in all_metrics]),
            'p95_latency_ms': np.percentile([m.latency_ms for m in all_metrics], 95),
            'p99_latency_ms': np.percentile([m.latency_ms for m in all_metrics], 99),
            'min_latency_ms': np.min([m.latency_ms for m in all_metrics]),
            'max_latency_ms': np.max([m.latency_ms for m in all_metrics]),
        }
        
        # Print results
        print(f"\n📊 Results for {search_name}:")
        print(f"  Precision@5:  {aggregate['avg_precision_at_5']:.3f}")
        print(f"  Precision@10: {aggregate['avg_precision_at_10']:.3f}")
        print(f"  Recall@10:    {aggregate['avg_recall_at_10']:.3f}")
        print(f"  Recall@20:    {aggregate['avg_recall_at_20']:.3f}")
        print(f"  NDCG@10:      {aggregate['avg_ndcg_at_10']:.3f}")
        print(f"  MRR:          {aggregate['avg_mrr']:.3f}")
        print(f"  Avg Latency:  {aggregate['avg_latency_ms']:.1f}ms")
        print(f"  P95 Latency:  {aggregate['p95_latency_ms']:.1f}ms")
        print(f"  P99 Latency:  {aggregate['p99_latency_ms']:.1f}ms")
        
        return aggregate
    
    def compare_methods(self) -> List[Dict[str, Any]]:
        """Compare all search methods"""
        results = []
        
        # Test hybrid search
        results.append(self.run_evaluation(
            hybrid_search_papers_impl,
            search_name="Hybrid Search (BM25 + KNN)",
            limit=20
        ))
        
        # Test semantic search
        results.append(self.run_evaluation(
            semantic_search_papers_impl,
            search_name="Semantic Search (SPECTER)",
            limit=20
        ))
        
        # Test keyword search
        results.append(self.run_evaluation(
            keyword_search_papers_impl,
            search_name="Keyword Search (BM25)",
            limit=20
        ))
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save evaluation results to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'results': results
            }, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")


def create_sample_queries() -> BEIRBenchmark:
    """Create comprehensive test queries (fallback if BEIR fails)"""
    benchmark = BEIRBenchmark(use_beir=False)
    
    from app.services.elasticsearch import ElasticsearchService
    from app.core.config import settings
    
    es_service = ElasticsearchService(settings.elasticsearch_config)
    
    # Define comprehensive test queries covering different categories
    test_queries = [
        # Architecture & Models (Easy)
        ('transformer architecture', 'method', 'easy'),
        ('convolutional neural networks', 'method', 'easy'),
        ('recurrent neural networks', 'method', 'easy'),
        ('attention mechanism', 'method', 'easy'),
        
        # Algorithms & Optimization (Medium)
        ('gradient descent optimization', 'method', 'medium'),
        ('reinforcement learning algorithms', 'method', 'medium'),
        ('generative adversarial networks', 'method', 'medium'),
        ('transfer learning techniques', 'method', 'medium'),
        ('ensemble learning methods', 'method', 'medium'),
        
        # Applications (Medium)
        ('natural language processing', 'application', 'medium'),
        ('computer vision tasks', 'application', 'medium'),
        ('speech recognition systems', 'application', 'medium'),
        ('recommendation systems', 'application', 'medium'),
        
        # Theory & Analysis (Hard)
        ('generalization bounds', 'theory', 'hard'),
        ('convergence analysis', 'theory', 'hard'),
        ('sample complexity', 'theory', 'hard'),
        
        # Specific Techniques (Medium-Hard)
        ('batch normalization', 'method', 'medium'),
        ('dropout regularization', 'method', 'medium'),
        ('adversarial training', 'method', 'hard'),
        ('meta learning', 'method', 'hard'),
        
        # Data & Datasets (Medium)
        ('image classification datasets', 'dataset', 'medium'),
        ('benchmark evaluation', 'dataset', 'medium'),
        
        # Cross-domain (Hard)
        ('graph neural networks', 'method', 'hard'),
        ('few-shot learning', 'method', 'hard'),
        ('self-supervised learning', 'method', 'hard'),
    ]
    
    print(f"Creating comprehensive benchmark with {len(test_queries)} queries...")
    
    for query_text, category, difficulty in test_queries:
        # Get relevant papers using hybrid search as ground truth
        results = es_service.hybrid_search(query_text, limit=10)
        relevant_ids = [r['id'] for r in results if 'id' in r][:5]  # Top 5 as ground truth
        
        if relevant_ids:
            benchmark.add_query(
                query=query_text,
                relevant_ids=relevant_ids,
                category=category,
                difficulty=difficulty
            )
    
    print(f"✅ Created {len(benchmark.queries)} test queries")
    return benchmark


def main():
    """Run BEIR SCIDOCS evaluation"""
    print("="*70)
    print("📊 BEIR SCIDOCS Benchmark Evaluation")
    print("="*70)
    
    # Use official BEIR dataset
    benchmark = BEIRBenchmark(dataset_name="scidocs", use_beir=True)
    
    # Fallback if BEIR fails
    if not benchmark.use_beir or len(benchmark.queries) == 0:
        print("⚠️  Using fallback manual queries...")
        benchmark = create_sample_queries()
    
    # Run comparison
    results = benchmark.compare_methods()
    
    # Save results
    output_file = 'eval/results/beir_scidocs_results.json' if benchmark.use_beir else 'eval/results/asta_benchmark_results.json'
    benchmark.save_results(results, output_file)
    
    print("\n" + "="*70)
    print("🎉 Evaluation Complete!")
    if benchmark.use_beir:
        print(f"📊 Official BEIR SCIDOCS: {len(benchmark.queries)} queries")
        print(f"💾 Results saved to: {output_file}")
    else:
        print(f"📊 Fallback mode: {len(benchmark.queries)} queries")
    print("="*70)


if __name__ == "__main__":
    main()
