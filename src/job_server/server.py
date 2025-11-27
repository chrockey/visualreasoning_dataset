"""
Job Distribution Server - FastAPI + SQLite

A simple, self-hosted job queue server for distributed processing.
Jobs are persistent and survive server restarts.

Usage:
    uvicorn job_server.server:app --host 0.0.0.0 --port 8000
"""

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Database path
DB_PATH = Path(__file__).parent / "jobs.db"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobSubmission(BaseModel):
    job_id: str
    payload: dict  # Any JSON-serializable data


class JobUpdate(BaseModel):
    status: JobStatus
    error_message: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    payload: dict
    status: JobStatus
    worker_id: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None


@contextmanager
def get_db():
    """Get database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize the database schema."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                worker_id TEXT,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                error_message TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")
        conn.commit()


# Initialize FastAPI app
app = FastAPI(title="Job Distribution Server")


@app.on_event("startup")
def startup():
    init_db()


@app.post("/jobs", response_model=JobResponse)
def submit_job(job: JobSubmission):
    """Submit a new job to the queue."""
    import json

    with get_db() as conn:
        try:
            conn.execute(
                """
                INSERT INTO jobs (job_id, payload, status, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (job.job_id, json.dumps(job.payload), JobStatus.PENDING, time.time()),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail=f"Job {job.job_id} already exists")

        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job.job_id,)).fetchone()

    return _row_to_response(row)


@app.post("/jobs/batch")
def submit_jobs_batch(jobs: list[JobSubmission]):
    """Submit multiple jobs at once."""
    import json

    submitted = []
    with get_db() as conn:
        for job in jobs:
            try:
                conn.execute(
                    """
                    INSERT INTO jobs (job_id, payload, status, created_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (job.job_id, json.dumps(job.payload), JobStatus.PENDING, time.time()),
                )
                submitted.append(job.job_id)
            except sqlite3.IntegrityError:
                pass  # Skip duplicates
        conn.commit()

    return {"submitted": len(submitted), "job_ids": submitted}


@app.get("/jobs/fetch")
def fetch_job(worker_id: str) -> Optional[JobResponse]:
    """
    Fetch a pending job and mark it as running.
    Returns null if no jobs are available.
    """
    import json

    with get_db() as conn:
        # Atomically claim a job
        row = conn.execute(
            """
            UPDATE jobs
            SET status = ?, worker_id = ?, started_at = ?
            WHERE job_id = (
                SELECT job_id FROM jobs
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT 1
            )
            RETURNING *
        """,
            (JobStatus.RUNNING, worker_id, time.time(), JobStatus.PENDING),
        ).fetchone()
        conn.commit()

    if row is None:
        return None

    return _row_to_response(row)


@app.put("/jobs/{job_id}")
def update_job(job_id: str, update: JobUpdate) -> JobResponse:
    """Update job status (complete or fail)."""
    with get_db() as conn:
        completed_at = time.time() if update.status in (JobStatus.COMPLETED, JobStatus.FAILED) else None

        conn.execute(
            """
            UPDATE jobs
            SET status = ?, completed_at = ?, error_message = ?
            WHERE job_id = ?
        """,
            (update.status, completed_at, update.error_message, job_id),
        )
        conn.commit()

        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return _row_to_response(row)


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    """Get job details by ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return _row_to_response(row)


@app.get("/stats")
def get_stats():
    """Get job queue statistics."""
    with get_db() as conn:
        stats = {}
        for status in JobStatus:
            count = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)
            ).fetchone()[0]
            stats[status.value] = count

        stats["total"] = sum(stats.values())

    return stats


@app.post("/jobs/{job_id}/retry")
def retry_job(job_id: str) -> JobResponse:
    """Reset a failed job back to pending status."""
    with get_db() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, worker_id = NULL, started_at = NULL,
                completed_at = NULL, error_message = NULL
            WHERE job_id = ? AND status = ?
        """,
            (JobStatus.PENDING, job_id, JobStatus.FAILED),
        )
        conn.commit()

        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return _row_to_response(row)


@app.post("/jobs/retry-all-failed")
def retry_all_failed():
    """Reset all failed jobs back to pending status."""
    with get_db() as conn:
        result = conn.execute(
            """
            UPDATE jobs
            SET status = ?, worker_id = NULL, started_at = NULL,
                completed_at = NULL, error_message = NULL
            WHERE status = ?
        """,
            (JobStatus.PENDING, JobStatus.FAILED),
        )
        conn.commit()
        count = result.rowcount

    return {"retried": count}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete a job."""
    with get_db() as conn:
        result = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {"deleted": job_id}


def _row_to_response(row) -> JobResponse:
    """Convert database row to response model."""
    import json

    return JobResponse(
        job_id=row["job_id"],
        payload=json.loads(row["payload"]),
        status=JobStatus(row["status"]),
        worker_id=row["worker_id"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        error_message=row["error_message"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
