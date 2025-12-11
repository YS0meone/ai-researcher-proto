"""Build arXiv ID mapping from local metadata."""
import json
from pathlib import Path
from tqdm import tqdm
import sys

def normalize_title(title: str) -> str:
    """Normalize title for matching."""
    return title.lower().strip().replace('\n', ' ').replace('\r', '')

def build_mapping():
    """Build title -> arXiv ID mapping from local metadata."""
    base_dir = Path(__file__).parent.parent
    metadata_file = base_dir / "papers" / "arxiv-metadata-oai-snapshot.json"
    corpus_file = base_dir / "eval" / "data" / "scidocs" / "corpus.jsonl"
    output_file = base_dir / "eval" / "data" / "scidocs" / "arxiv_id_mapping.json"
    
    print(f"Loading arXiv metadata from {metadata_file}...")
    title_to_arxiv_id = {}
    
    # Build mapping from arXiv metadata (title -> arXiv ID)
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"Processed {i:,} arXiv papers...")
            
            try:
                paper = json.loads(line)
                title_norm = normalize_title(paper['title'])
                arxiv_id = paper['id']
                title_to_arxiv_id[title_norm] = arxiv_id
            except Exception as e:
                continue
    
    print(f"\nBuilt mapping with {len(title_to_arxiv_id):,} arXiv papers")
    
    # Match BEIR corpus titles to arXiv IDs
    print(f"\nMatching BEIR corpus from {corpus_file}...")
    corpus_to_arxiv = {}
    matched = 0
    total = 0
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Matching"):
            total += 1
            doc = json.loads(line)
            corpus_id = doc['_id']
            title_norm = normalize_title(doc['title'])
            
            if title_norm in title_to_arxiv_id:
                corpus_to_arxiv[corpus_id] = title_to_arxiv_id[title_norm]
                matched += 1
    
    print(f"\nMatched: {matched}/{total} ({matched/total*100:.1f}%)")
    
    # Save mapping
    print(f"Saving mapping to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(corpus_to_arxiv, f, indent=2)
    
    print(f"Done! Mapping saved with {len(corpus_to_arxiv):,} entries")
    
    # Show sample mappings
    print("\nSample mappings:")
    for i, (corpus_id, arxiv_id) in enumerate(list(corpus_to_arxiv.items())[:5]):
        print(f"  {corpus_id[:40]}... -> {arxiv_id}")

if __name__ == "__main__":
    build_mapping()
