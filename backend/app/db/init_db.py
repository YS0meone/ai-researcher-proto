import asyncio
from urllib.parse import urlparse
from sqlalchemy import create_engine
from app.core.config import settings
from app.db.models import Base

tmpPostgres = urlparse(settings.DATABASE_URL)


def init_db() -> None:
    engine = create_engine(settings.DATABASE_URL, echo=settings.DATABASE_SQL_ECHO)
    Base.metadata.create_all(engine)
    print("Database initialized")

if __name__ == "__main__":
    init_db()