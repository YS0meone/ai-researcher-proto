from app.core.config import settings
from app.services.paper_loader import PaperLoader

    
def main():
    loader = PaperLoader(settings.paper_loader_config)
    loader.load_by_metadata(categories_filter=['cs.CL'], limit=10)
    

if __name__ == "__main__":
    main()