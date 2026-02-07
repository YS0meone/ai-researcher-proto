from fastapi import FastAPI
from pydantic import BaseModel

from app.db.schema import S2Paper
from app.tasks.ingest import ingest_paper_task
from app.celery_app import celery_app

app = FastAPI()


# ---------------------
# Request / Response models
# ---------------------

class IngestRequest(BaseModel):
    """POST body for /ingest."""
    papers: list[S2Paper]


class TaskRef(BaseModel):
    paperId: str
    taskId: str


class IngestResponse(BaseModel):
    """Returned immediately from POST /ingest."""
    tasks: list[TaskRef]


class TaskStatus(BaseModel):
    taskId: str
    state: str  # PENDING | STARTED | SUCCESS | FAILURE
    result: dict | None = None


# ---------------------
# Endpoints
# ---------------------

@app.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_papers(req: IngestRequest):
    """Accept a list of S2Paper objects and dispatch one Celery task per paper.

    Returns 202 Accepted with a list of {paperId, taskId} so the caller can
    poll for status later.
    """
    tasks: list[TaskRef] = []
    for paper in req.papers:
        # Serialize to JSON-safe dict before sending to Celery
        paper_dict = paper.model_dump(mode="json")
        result = ingest_paper_task.delay(paper_dict)
        tasks.append(TaskRef(paperId=paper.paperId, taskId=result.id))

    return IngestResponse(tasks=tasks)


@app.get("/ingest/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Check the status of a single ingestion task."""
    result = celery_app.AsyncResult(task_id)
    return TaskStatus(
        taskId=task_id,
        state=result.state,
        result=result.result if result.ready() else None,
    )


@app.post("/ingest/status/batch")
async def get_batch_status(task_ids: list[str]):
    """Check the status of multiple ingestion tasks at once."""
    statuses: list[TaskStatus] = []
    for task_id in task_ids:
        result = celery_app.AsyncResult(task_id)
        statuses.append(
            TaskStatus(
                taskId=task_id,
                state=result.state,
                result=result.result if result.ready() else None,
            )
        )
    return {"statuses": statuses}
