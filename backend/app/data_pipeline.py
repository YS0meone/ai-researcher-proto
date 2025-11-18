from app.core.config import settings
from app.services.paper_loader import PaperLoader
import sys

def main():
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    
    print("="*60)
    print("🔬 AI Research Paper Loader - Parallel Processing")
    print("="*60)
    
    loader = PaperLoader(settings.paper_loader_config)
    
    # Load all Computer Science papers (any cs.* category)
    # ArXiv has ~40 different cs.* categories
    cs_categories = [
        'cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL', 'cs.CR', 'cs.CV',
        'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL',
        'cs.GL', 'cs.GR', 'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO',
        'cs.MA', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS',
        'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY'
    ]
    
    print(f"\n📋 Configuration:")
    print(f"   Including all CS categories: {len(cs_categories)} categories")
    print(f"   Expected match rate: ~20-30% of ArXiv papers")
    print()
    
    # Process enough lines to get a good dataset
    limit = 10000  # Process 500k lines to get ~100k-150k CS papers
    
    # PDF Processing: Controlled by LOADER_PROCESS_PDFS env variable
    # False (default) = Fast bulk loading (metadata + title/abstract embeddings only)
    # True = Full-text search (downloads PDFs and parses with GROBID - 20x slower!)
    process_pdfs = settings.paper_loader_config.process_pdfs
    
    print(f"⚡ Speed mode: {'SLOW - Full PDFs' if process_pdfs else 'FAST - Metadata only'}")
    print()
    
    result = loader.load_by_metadata_parallel(
        categories_filter=cs_categories,
        limit=limit,
        process_pdfs=process_pdfs
    )
    
    print(f"\n{'='*60}")
    print(f"📈 Final Statistics")
    print(f"{'='*60}")
    print(f"✅ Processed: {result['total_processed']:,} papers")
    print(f"❌ Errors: {result['total_errors']:,}")
    print(f"⏱️  Duration: {result['duration']:.2f}s ({result['duration']/60:.1f} min)")
    print(f"⚡ Rate: {result['rate']:.2f} papers/sec")
    
    # Calculate match rate
    if result['total_processed'] > 0 and limit > 0:
        match_rate = result['total_processed'] / limit * 100
        print(f"📊 Match rate: {match_rate:.1f}% of lines contained CS papers")
    
    # Memory and performance notes
    print(f"\n{'='*60}")
    print(f"⚙️  Configuration")
    print(f"{'='*60}")
    print(f"👷 Workers: {settings.paper_loader_config.workers}")
    print(f"📦 Batch size: {settings.paper_loader_config.batch_size}")
    print(f"🏷️  CS Categories: {len(cs_categories)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

