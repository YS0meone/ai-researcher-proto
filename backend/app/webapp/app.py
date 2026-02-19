import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.schema import S2Paper
from app.tasks.ingest import ingest_paper_task
from app.celery_app import celery_app

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _send_ingest_task(paper_dict: dict) -> str:
    """Synchronous helper — keeps Celery proxy resolution off the event loop.
    Returns only the task ID string so the AsyncResult is never passed to the event loop."""
    async_result = ingest_paper_task.delay(paper_dict)
    task_id = async_result.id
    del async_result  # ensure __del__ fires here in the thread, not on the event loop
    return task_id


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
        task_id = await asyncio.to_thread(_send_ingest_task, paper_dict)
        tasks.append(TaskRef(paperId=paper.paperId, taskId=task_id))

    return IngestResponse(tasks=tasks)


def _check_task(task_id: str) -> TaskStatus:
    """Synchronous helper — safe to run in a thread."""
    result = celery_app.AsyncResult(task_id)
    res = result.result if result.ready() else None
    if res is not None and not isinstance(res, dict):
        res = {"error": str(res)}
    status = TaskStatus(taskId=task_id, state=result.state, result=res)
    del result  # ensure __del__ fires here in the thread, not on the event loop
    return status


@app.get("/ingest/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Check the status of a single ingestion task."""
    return await asyncio.to_thread(_check_task, task_id)


class BatchStatusRequest(BaseModel):
    task_ids: list[str]


@app.post("/ingest/status/batch")
async def get_batch_status(req: BatchStatusRequest):
    """Check the status of multiple ingestion tasks at once."""
    statuses = [await asyncio.to_thread(_check_task, tid) for tid in req.task_ids]
    return {"statuses": statuses}
