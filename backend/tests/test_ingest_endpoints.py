from unittest.mock import MagicMock, patch

import pytest


class TestPostIngest:
    """Tests for POST /ingest endpoint."""

    @pytest.fixture
    def mock_delay(self):
        """Patch ingest_paper_task.delay and return the mock."""
        with patch("app.webapp.app.ingest_paper_task") as mock_task:
            mock_result = MagicMock()
            mock_result.id = "task-abc-123"
            mock_task.delay.return_value = mock_result
            yield mock_task

    async def test_ingest_single_paper(self, client, mock_delay):
        payload = {"papers": [{"paperId": "abc123"}]}
        resp = await client.post("/ingest", json=payload)

        assert resp.status_code == 202
        body = resp.json()
        assert len(body["tasks"]) == 1
        assert body["tasks"][0]["paperId"] == "abc123"
        assert body["tasks"][0]["taskId"] == "task-abc-123"
        mock_delay.delay.assert_called_once()
        call_arg = mock_delay.delay.call_args[0][0]
        assert call_arg["paperId"] == "abc123"

    async def test_ingest_multiple_papers(self, client):
        ids = [f"paper-{i}" for i in range(3)]
        task_ids = [f"task-{i}" for i in range(3)]

        with patch("app.webapp.app.ingest_paper_task") as mock_task:
            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                m = MagicMock()
                m.id = task_ids[call_count]
                call_count += 1
                return m

            mock_task.delay.side_effect = side_effect

            payload = {"papers": [{"paperId": pid} for pid in ids]}
            resp = await client.post("/ingest", json=payload)

        assert resp.status_code == 202
        tasks = resp.json()["tasks"]
        assert len(tasks) == 3
        for i, t in enumerate(tasks):
            assert t["paperId"] == ids[i]
            assert t["taskId"] == task_ids[i]

    async def test_ingest_empty_list(self, client, mock_delay):
        resp = await client.post("/ingest", json={"papers": []})
        assert resp.status_code == 202
        assert resp.json()["tasks"] == []
        mock_delay.delay.assert_not_called()

    async def test_ingest_missing_paper_id(self, client, mock_delay):
        # paperId is required on S2Paper
        resp = await client.post("/ingest", json={"papers": [{"title": "no id"}]})
        assert resp.status_code == 422

    async def test_ingest_missing_papers_field(self, client, mock_delay):
        resp = await client.post("/ingest", json={"not_papers": []})
        assert resp.status_code == 422

    async def test_ingest_paper_with_all_optional_fields(self, client, mock_delay):
        paper = {
            "paperId": "full-paper",
            "abstract": "An abstract",
            "authors": [{"name": "Alice"}],
            "citationCount": 42,
            "corpusId": 999,
            "externalIds": {"DOI": "10.1234/test"},
            "fieldsOfStudy": ["Computer Science"],
            "influentialCitationCount": 5,
            "isOpenAccess": True,
            "journal": {"name": "Nature", "volume": "1", "pages": "1-10"},
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
            "publicationDate": "2024-01-15T00:00:00",
            "publicationTypes": ["JournalArticle"],
            "referenceCount": 10,
            "s2FieldsOfStudy": [{"category": "CS", "source": "s2"}],
            "title": "A Great Paper",
            "url": "https://semanticscholar.org/paper/full-paper",
            "venue": "NeurIPS",
            "year": 2024,
            "tldr": {"text": "A short summary"},
        }
        resp = await client.post("/ingest", json={"papers": [paper]})
        assert resp.status_code == 202
        call_arg = mock_delay.delay.call_args[0][0]
        assert call_arg["paperId"] == "full-paper"
        assert call_arg["abstract"] == "An abstract"
        assert call_arg["citationCount"] == 42
        assert call_arg["journal"]["name"] == "Nature"


# ---------------------------------------------------------------------------
# GET /ingest/status/{task_id}
# ---------------------------------------------------------------------------


class TestGetStatus:
    """Tests for GET /ingest/status/{task_id} endpoint."""

    def _mock_async_result(self, state, result=None, ready=False):
        mock = MagicMock()
        mock.state = state
        mock.result = result
        mock.ready.return_value = ready
        return mock

    async def test_status_pending(self, client):
        with patch("app.webapp.app.celery_app") as mock_celery:
            mock_celery.AsyncResult.return_value = self._mock_async_result(
                "PENDING"
            )
            resp = await client.get("/ingest/status/task-1")

        assert resp.status_code == 200
        body = resp.json()
        assert body["taskId"] == "task-1"
        assert body["state"] == "PENDING"
        assert body["result"] is None

    async def test_status_started(self, client):
        with patch("app.webapp.app.celery_app") as mock_celery:
            mock_celery.AsyncResult.return_value = self._mock_async_result(
                "STARTED"
            )
            resp = await client.get("/ingest/status/task-2")

        body = resp.json()
        assert body["state"] == "STARTED"
        assert body["result"] is None

    async def test_status_success(self, client):
        result_data = {"ingested": True, "paperId": "abc"}
        with patch("app.webapp.app.celery_app") as mock_celery:
            mock_celery.AsyncResult.return_value = self._mock_async_result(
                "SUCCESS", result=result_data, ready=True
            )
            resp = await client.get("/ingest/status/task-3")

        body = resp.json()
        assert body["state"] == "SUCCESS"
        assert body["result"] == result_data

    async def test_status_failure(self, client):
        error_info = {"error": "Something went wrong"}
        with patch("app.webapp.app.celery_app") as mock_celery:
            mock_celery.AsyncResult.return_value = self._mock_async_result(
                "FAILURE", result=error_info, ready=True
            )
            resp = await client.get("/ingest/status/task-4")

        body = resp.json()
        assert body["state"] == "FAILURE"
        assert body["result"] == error_info


# ---------------------------------------------------------------------------
# POST /ingest/status/batch
# ---------------------------------------------------------------------------


class TestBatchStatus:
    """Tests for POST /ingest/status/batch endpoint."""

    def _mock_async_result(self, state, result=None, ready=False):
        mock = MagicMock()
        mock.state = state
        mock.result = result
        mock.ready.return_value = ready
        return mock

    async def test_batch_multiple_tasks(self, client):
        results_map = {
            "t1": self._mock_async_result("PENDING"),
            "t2": self._mock_async_result("SUCCESS", result={"ok": True}, ready=True),
            "t3": self._mock_async_result("FAILURE", result={"err": "x"}, ready=True),
        }
        with patch("app.webapp.app.celery_app") as mock_celery:
            mock_celery.AsyncResult.side_effect = lambda tid: results_map[tid]
            resp = await client.post(
                "/ingest/status/batch", json=["t1", "t2", "t3"]
            )

        assert resp.status_code == 200
        statuses = resp.json()["statuses"]
        assert len(statuses) == 3
        assert statuses[0]["state"] == "PENDING"
        assert statuses[0]["result"] is None
        assert statuses[1]["state"] == "SUCCESS"
        assert statuses[1]["result"] == {"ok": True}
        assert statuses[2]["state"] == "FAILURE"
        assert statuses[2]["result"] == {"err": "x"}

    async def test_batch_empty_list(self, client):
        with patch("app.webapp.app.celery_app"):
            resp = await client.post("/ingest/status/batch", json=[])

        assert resp.status_code == 200
        assert resp.json()["statuses"] == []

    async def test_batch_single_task(self, client):
        with patch("app.webapp.app.celery_app") as mock_celery:
            mock_celery.AsyncResult.return_value = self._mock_async_result(
                "STARTED"
            )
            resp = await client.post(
                "/ingest/status/batch", json=["only-one"]
            )

        statuses = resp.json()["statuses"]
        assert len(statuses) == 1
        assert statuses[0]["taskId"] == "only-one"
        assert statuses[0]["state"] == "STARTED"
