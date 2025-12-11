"""Extract matched papers from arXiv metadata and create SCIDOCS subset."""
import json
from pathlib import Path
from tqdm import tqdm

def extract_matched_papers():
    """Extract papers that matched between BEIR and arXiv."""
    base_dir = Path(__file__).parent.parent
    
    # Load mapping
    mapping_file = base_dir / "eval" / "data" / "scidocs" / "arxiv_id_mapping.json"
    print(f"Loading mapping from {mapping_file}...")
    with open(mapping_file, 'r') as f:
        corpus_to_arxiv = json.load(f)
    
    print(f"Loaded {len(corpus_to_arxiv):,} mappings")
    
    # Build reverse mapping (arxiv_id -> corpus_id)
    arxiv_to_corpus = {v: k for k, v in corpus_to_arxiv.items()}
    print(f"Built reverse mapping: {len(arxiv_to_corpus):,} arXiv IDs")
    
    # Extract papers from arXiv metadata
    metadata_file = base_dir / "papers" / "arxiv-metadata-oai-snapshot.json"
    output_file = base_dir / "eval" / "data" / "scidocs" / "arxiv_matched_papers.jsonl"
    
    print(f"\nExtracting papers from {metadata_file}...")
    matched_papers = {}
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"Processed {i:,} papers, found {len(matched_papers):,} matches...")
            
            try:
                paper = json.loads(line)
                arxiv_id = paper['id']
                
                if arxiv_id in arxiv_to_corpus:
                    corpus_id = arxiv_to_corpus[arxiv_id]
                    matched_papers[corpus_id] = {
                        'arxiv_id': arxiv_id,
                        'title': paper['title'],
                        'abstract': paper['abstract'],
                        'authors': paper.get('authors', ''),
                        'categories': paper.get('categories', ''),
                        'year': paper.get('versions', [{}])[0].get('created', '')[:4] if paper.get('versions') else None
                    }
                    
                    # Stop if we found all matches
                    if len(matched_papers) == len(corpus_to_arxiv):
                        print(f"\nFound all {len(matched_papers):,} matched papers!")
                        break
                        
            except Exception as e:
                continue
    
    print(f"\nTotal matched papers found: {len(matched_papers):,}/{len(corpus_to_arxiv):,}")
    
    # Save extracted papers
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for corpus_id, paper_data in matched_papers.items():
            f.write(json.dumps({
                'corpus_id': corpus_id,
                **paper_data
            }) + '\n')
    
    print(f"Done! Saved {len(matched_papers):,} papers")
    
    # Create filtered SCIDOCS subset
    print("\nCreating SCIDOCS subset with matched papers only...")
    create_scidocs_subset(matched_papers, base_dir)
    
    return matched_papers

def create_scidocs_subset(matched_papers, base_dir):
    """Create SCIDOCS subset with only matched papers."""
    matched_corpus_ids = set(matched_papers.keys())
    
    # Filter corpus
    print("Filtering corpus.jsonl...")
    corpus_in = base_dir / "eval" / "data" / "scidocs" / "corpus.jsonl"
    corpus_out = base_dir / "eval" / "data" / "scidocs" / "corpus_matched.jsonl"
    
    count = 0
    with open(corpus_in, 'r', encoding='utf-8') as fin, \
         open(corpus_out, 'w', encoding='utf-8') as fout:
        for line in fin:
            doc = json.loads(line)
            if doc['_id'] in matched_corpus_ids:
                fout.write(line)
                count += 1
    
    print(f"  Saved {count:,} corpus documents")
    
    # Filter queries (only keep queries that have matched papers in qrels)
    print("Filtering queries.jsonl...")
    qrels_file = base_dir / "eval" / "data" / "scidocs" / "qrels" / "test.tsv"
    
    # Load qrels to see which queries are relevant
    relevant_queries = set()
    with open(qrels_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id = parts[0]
                corpus_id = parts[1]
                score = int(parts[2])
                if corpus_id in matched_corpus_ids and score > 0:
                    relevant_queries.add(query_id)
    
    print(f"  Found {len(relevant_queries):,} queries with matched papers")
    
    queries_in = base_dir / "eval" / "data" / "scidocs" / "queries.jsonl"
    queries_out = base_dir / "eval" / "data" / "scidocs" / "queries_matched.jsonl"
    
    count = 0
    with open(queries_in, 'r', encoding='utf-8') as fin, \
         open(queries_out, 'w', encoding='utf-8') as fout:
        for line in fin:
            query = json.loads(line)
            if query['_id'] in relevant_queries:
                fout.write(line)
                count += 1
    
    print(f"  Saved {count:,} queries")
    
    # Filter qrels
    print("Filtering qrels/test.tsv...")
    qrels_out = base_dir / "eval" / "data" / "scidocs" / "qrels" / "test_matched.tsv"
    
    count = 0
    with open(qrels_file, 'r') as fin, \
         open(qrels_out, 'w') as fout:
        header = next(fin)
        fout.write(header)
        
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id = parts[0]
                corpus_id = parts[1]
                if query_id in relevant_queries and corpus_id in matched_corpus_ids:
                    fout.write(line)
                    count += 1
    
    print(f"  Saved {count:,} relevance judgments")
    
    print("\nSCIDOCS matched subset created:")
    print(f"  - {len(matched_corpus_ids):,} papers")
    print(f"  - {len(relevant_queries):,} queries")
    print(f"  - {count:,} relevance judgments")
    print("\nFiles created:")
    print(f"  - corpus_matched.jsonl")
    print(f"  - queries_matched.jsonl")
    print(f"  - qrels/test_matched.tsv")

if __name__ == "__main__":
    extract_matched_papers()
