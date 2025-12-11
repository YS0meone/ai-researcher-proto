# Paper Finder Evaluation

Two-layer benchmark system for academic paper retrieval.

## Quick Start

**Retrieval System Benchmark (BEIR SCIDOCS)**
```bash
python eval/beir_matched_evaluation.py  # Matched subset evaluation
```

**AI Agent Evaluation (ASTA-bench)**
```bash
# 1. Install ASTA-bench framework
git clone --recursive https://github.com/allenai/asta-bench.git
cd asta-bench && make shell

# 2. Copy our solver
cp /path/to/astabench_integration.py solvers/paper_finder.py

# 3. Run evaluation
export ASTA_TOOL_KEY=<key> DEEPSEEK_API_KEY=<key>
uv run astabench eval --solver paper_finder_agent --split validation
```

See `EVALUATION_REPORT.md` for detailed results.

---

## Files

```
eval/
├── beir_matched_evaluation.py    # BEIR benchmark with ID mapping
├── beir_retrieval_benchmark.py   # Full BEIR SCIDOCS benchmark
├── astabench_integration.py      # ASTA-bench solver wrapper
├── build_id_mapping.py           # ArXiv ID to paper mapping builder
├── extract_matched_papers.py     # Extract matched subset from corpus
├── monitor_eval.py               # Evaluation monitoring utility
├── load_qasper.py                # QASPER dataset loader
├── EVALUATION_REPORT.md          # Detailed evaluation results
├── data/scidocs/                 # BEIR SCIDOCS matched subset (gitignored)
└── results/
    └── beir_matched_results.json # Benchmark results
```

---

## References

- [ASTA-bench](https://github.com/allenai/asta-bench) - AI Agent Evaluation
- [BEIR](https://github.com/beir-cellar/beir) - Information Retrieval Benchmark
- [InspectAI](https://inspect.ai-safety-institute.org.uk/) - Evaluation Framework
