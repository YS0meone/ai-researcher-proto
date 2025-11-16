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
    
    # Scale test with larger dataset
    result = loader.load_by_metadata_parallel(
        categories_filter=['cs.CL', 'cs.AI', 'cs.LG'],  # Broader filter
        limit=100000  # Process 1000 lines
    )
    
    print(f"\n{'='*60}")
    print(f"📈 Final Statistics")
    print(f"{'='*60}")
    print(f"✅ Processed: {result['total_processed']:,} papers")
    print(f"❌ Errors: {result['total_errors']:,}")
    print(f"⏱️  Duration: {result['duration']:.2f}s ({result['duration']/60:.1f} min)")
    print(f"⚡ Rate: {result['rate']:.2f} papers/sec")
    
    # Memory and performance notes
    print(f"\n{'='*60}")
    print(f"⚙️  Configuration")
    print(f"{'='*60}")
    print(f"👷 Workers: {settings.paper_loader_config.workers}")
    print(f"📦 Batch size: {settings.paper_loader_config.batch_size}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

