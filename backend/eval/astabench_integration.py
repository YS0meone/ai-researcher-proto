"""
ASTA-bench Integration for Paper Finder Agent

Wraps our LangGraph agent to work with AllenAI's ASTA-bench framework.
See: https://github.com/allenai/asta-bench

Usage:
    # 1. Install ASTA-bench
    git clone --recursive https://github.com/allenai/asta-bench.git
    cd asta-bench && make shell
    
    # 2. Copy this file to asta-bench/solvers/
    cp astabench_integration.py /path/to/asta-bench/solvers/paper_finder.py
    
    # 3. Run evaluation
    export ASTA_TOOL_KEY=<key>
    export OPENAI_API_KEY=<key>
    uv run astabench eval --solver paper_finder_agent --split validation
"""

import json
import logging
from typing import Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Import dependencies (will be available in ASTA-bench environment)
try:
    from inspect_ai import solver
    from inspect_ai.solver import TaskState
except ImportError:
    logger.warning("InspectAI not installed. This file requires ASTA-bench setup.")
    solver = None

# Import our agent (adjust path if needed)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.agent.graph import graph
except ImportError as e:
    logger.warning(f"Agent import failed: {e}")
    graph = None


# ============================================================================
# ASTA-bench Solver
# ============================================================================

def paper_finder_agent(**kwargs):
    """
    InspectAI solver for ASTA-bench evaluation.
    
    Wraps our LangGraph agent to comply with ASTA-bench's interface.
    
    Example:
        uv run astabench eval --solver paper_finder_agent \\
            --split validation astabench/paper_finder_validation
    """
    if not solver:
        raise ImportError("InspectAI not installed. Run from ASTA-bench environment.")
    
    @solver
    def solve():
        async def run_agent(state: TaskState, generate_fn):
            query = state.input
            logger.info(f"Query: {query}")
            
            try:
                # Run our LangGraph agent
                result = await graph.ainvoke({
                    "messages": [{"role": "user", "content": query}],
                    "papers": [],
                })
                
                # Extract papers from agent result
                papers = result.get("papers", [])
                
                # Format for ASTA-bench (Semantic Scholar corpus_id format)
                output = {
                    "output": {
                        "results": [
                            {
                                "paper_id": extract_corpus_id(paper),
                                "markdown_evidence": format_evidence(paper)
                            }
                            for paper in papers[:250]  # ASTA-bench limit
                        ]
                    }
                }
                
                state.output.completion = json.dumps(output, indent=2)
                logger.info(f"Found {len(papers)} papers")
                
            except Exception as e:
                logger.error(f"Agent failed: {e}")
                state.output.completion = json.dumps({
                    "output": {"results": []},
                    "error": str(e)
                })
            
            return state
        
        return run_agent
    
    return solve()


def extract_corpus_id(paper: Dict[str, Any]) -> str:
    """Extract Semantic Scholar corpus_id from paper dict."""
    # Try different possible fields
    for field in ["corpus_id", "corpusId", "id", "paper_id"]:
        if value := paper.get(field):
            return str(value)
    return ""


def format_evidence(paper: Dict[str, Any]) -> str:
    """Format paper as markdown evidence for ASTA-bench."""
    title = paper.get("title", "Unknown")
    year = paper.get("year", "")
    abstract = paper.get("abstract", "")
    
    # ASTA-bench requires title and year at start
    evidence = f"**{title}**"
    if year:
        evidence += f" ({year})"
    
    if abstract:
        evidence += f"\n\n{abstract[:1000]}"  # Limit length
    
    return evidence


# ============================================================================
# Standalone Testing (Optional)
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║          ASTA-bench Integration - Paper Finder           ║
╚══════════════════════════════════════════════════════════╝

Setup Instructions:
1. Install ASTA-bench framework:
   git clone --recursive https://github.com/allenai/asta-bench.git
   cd asta-bench && make shell

2. Copy this file to ASTA-bench:
   cp astabench_integration.py /path/to/asta-bench/solvers/paper_finder.py

3. Configure environment:
   export ASTA_TOOL_KEY=<your-key>
   export OPENAI_API_KEY=<your-key>
   export HF_TOKEN=<huggingface-token>

4. Run evaluation:
   uv run astabench eval --solver paper_finder_agent \\
       --split validation \\
       astabench/paper_finder_validation

Documentation: https://github.com/allenai/asta-bench
""")