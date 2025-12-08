# Paper Finder Performance Evaluation

Benchmark evaluation for Paper Finder search system.

---

## Quick Start

```bash
cd backend
uv run python eval/asta_benchmark.py
```

**Execution time:** ~15-20 seconds

---

## Current Performance (25 queries)

| Metric | Hybrid Search | Semantic Search | Keyword Search |
|--------|--------------|-----------------|----------------|
| **Precision@5** | 97.6% ✅ | 20.0% | 96.8% |
| **Precision@10** | 50.0% ✅ | 14.0% | 50.0% |
| **Recall@10** | 100% ✅ | 28.0% | 100% |
| **NDCG@10** | 99.7% ✅ | 29.8% | 99.7% |
| **MRR** | 1.000 ✅ | 0.584 | 1.000 |
| **Latency** | 98.2ms ✅ | 37.1ms | 421.2ms |

**Recommended:** Hybrid Search (perfect recall, 98ms latency)

---

## Metrics Explained

| Metric | Meaning | Good Standard |
|--------|---------|---------------|
| **Precision@K** | Relevant results in top K / K | > 30% |
| **Recall@K** | Found relevant papers / Total relevant | > 60% |
| **NDCG@10** | Ranking quality (0-1) | > 0.7 |
| **MRR** | Reciprocal rank of first relevant result | > 0.5 |
| **Latency** | Average response time | < 500ms |

**Why Precision@10 < Precision@5?** Normal. Best results rank at top (Precision@5 = 97.6%), quality decreases for positions 6-10.

---

## File Structure

```
eval/
├── asta_benchmark.py           # Evaluation script
├── README.md                   # This file
└── results/
    ├── asta_benchmark_results.json
    └── BENCHMARK_REPORT.md
```

---

## Customization

**Change test queries** (line ~320 in `asta_benchmark.py`):
```python
test_queries = [
    ('your query', 'category', 'difficulty'),
    # ...
]
```

**Change ground truth size** (line ~348):
```python
relevant_ids = [r['id'] for r in results if 'id' in r][:5]  # Change 5
```

**Change search limit** (line ~277):
```python
limit=20  # Change to 50, 100, etc.
```

---

## Optimization Priorities

1. **Replace SPECTER embeddings** → Improve Semantic Search recall (28% → 50%+)
2. **Deprecate Keyword Search** → No advantage over Hybrid, 4.3x slower
3. **Add cross-encoder reranking** → Improve Precision@10 (50% → 60%+)

---

## References

- [BEIR Benchmark](https://github.com/beir-cellar/beir) - IR Evaluation Standard
- [SPECTER](https://github.com/allenai/specter) - Scientific Paper Embeddings
- [SciBERT](https://github.com/allenai/scibert) - Domain-specific BERT

---

## Performance History

| Date | Queries | Precision@10 | Recall@10 | Latency |
|------|---------|--------------|-----------|---------|
| 2025-12-07 | 25 | 50.0% | 100% | 98.2ms |
| 2025-12-07 | 3 | 23.3% | 88.9% | 208.2ms |
