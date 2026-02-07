import logging
from app.celery_app import celery_app
from app.db.schema import S2Paper
from app.core.config import settings
from app.services.qdrant import QdrantService

logger = logging.getLogger(__name__)

# Lazily initialized QdrantService (one per worker process)
_qdrant_service: QdrantService | None = None


def _get_qdrant_service() -> QdrantService:
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService(settings.qdrant_config)
    return _qdrant_service


@celery_app.task(bind=True, name="ingest_paper", max_retries=2)
def ingest_paper_task(self, paper_dict: dict) -> dict:
    """Celery task that ingests a single S2Paper into Qdrant.

    Attempts full-PDF ingestion first.  If no open-access PDF is available,
    falls back to abstract-only ingestion.

    Args:
        paper_dict: JSON-serializable dict representing an S2Paper.

    Returns:
        Status dict with paperId, method used, chunk_count, and success flag.
    """
    paper = S2Paper(**paper_dict)
    qdrant = _get_qdrant_service()

    # Try full PDF ingestion first
    has_pdf = bool(paper.openAccessPdf and paper.openAccessPdf.get("url"))

    if has_pdf:
        try:
            chunk_count = qdrant.add_s2_paper(paper)
            logger.info(f"Ingested paper {paper.paperId} via PDF ({chunk_count} chunks)")
            return {
                "paperId": paper.paperId,
                "method": "full_pdf",
                "chunk_count": chunk_count,
                "success": True,
            }
        except Exception as e:
            logger.warning(
                f"PDF ingestion failed for {paper.paperId}, falling back to abstract: {e}"
            )

    # Fallback: abstract-only
    try:
        chunk_count = qdrant.add_s2_paper_abstract_only(paper)
        logger.info(f"Ingested paper {paper.paperId} via abstract-only")
        return {
            "paperId": paper.paperId,
            "method": "abstract_only",
            "chunk_count": chunk_count,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Ingestion completely failed for {paper.paperId}: {e}")
        return {
            "paperId": paper.paperId,
            "method": "none",
            "chunk_count": 0,
            "success": False,
            "error": str(e),
        }
