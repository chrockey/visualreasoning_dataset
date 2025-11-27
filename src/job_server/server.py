"""
Job Distribution Server - FastAPI + SQLAlchemy + SQLAdmin

A simple, self-hosted job queue server for distributed processing.
Jobs are organized into experiments. Workers fetch jobs by experiment_id.

Usage:
    uvicorn src.job_server.server:app --host 0.0.0.0 --port 8000

Admin panel available at /admin
"""

import json
import secrets
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker
from sqladmin import Admin, ModelView

# Database setup
DB_PATH = Path(__file__).parent / "jobs.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Configuration
MAX_JOB_RETRIES = 3  # Maximum number of times a failed job can be retried

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30.0},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ============ Enums ============


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"


# ============ SQLAlchemy Models ============


class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    config_file = Column(String, nullable=True)
    status = Column(String, nullable=False, default=ExperimentStatus.ACTIVE.value)
    created_at = Column(Float, nullable=False)

    jobs = relationship("Job", back_populates="experiment", cascade="all, delete-orphan", lazy="dynamic")

    def __str__(self):
        return self.experiment_id

    @property
    def total_jobs(self) -> int:
        return self.jobs.count()

    @property
    def pending_jobs(self) -> int:
        return self.jobs.filter_by(status=JobStatus.PENDING.value).count()

    @property
    def running_jobs(self) -> int:
        return self.jobs.filter_by(status=JobStatus.RUNNING.value).count()

    @property
    def completed_jobs(self) -> int:
        return self.jobs.filter_by(status=JobStatus.COMPLETED.value).count()

    @property
    def failed_jobs(self) -> int:
        return self.jobs.filter_by(status=JobStatus.FAILED.value).count()

    @property
    def progress(self) -> str:
        total = self.total_jobs
        if total == 0:
            return "0/0"
        done = self.completed_jobs + self.failed_jobs
        pct = done / total * 100
        return f"{done}/{total} ({pct:.0f}%)"


class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey("experiments.experiment_id"), nullable=False)
    payload = Column(Text, nullable=False)
    status = Column(String, nullable=False, default=JobStatus.PENDING.value)
    worker_id = Column(String, nullable=True)
    created_at = Column(Float, nullable=False)
    started_at = Column(Float, nullable=True)
    completed_at = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    failure_count = Column(Integer, nullable=False, default=0)

    experiment = relationship("Experiment", back_populates="jobs")

    __table_args__ = (
        Index("idx_jobs_status", "status"),
        Index("idx_jobs_experiment", "experiment_id"),
        Index("idx_jobs_exp_status", "experiment_id", "status"),
    )

    def __str__(self):
        return self.job_id


# ============ Pydantic Models ============


class ExperimentCreate(BaseModel):
    experiment_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    config_file: Optional[str] = None
    num_jobs: Optional[int] = None


class ExperimentResponse(BaseModel):
    experiment_id: str
    name: Optional[str]
    description: Optional[str]
    config_file: Optional[str]
    status: ExperimentStatus
    created_at: float

    class Config:
        from_attributes = True


class JobSubmission(BaseModel):
    job_id: str
    payload: dict


class JobResponse(BaseModel):
    job_id: str
    experiment_id: str
    payload: dict
    status: JobStatus
    worker_id: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    failure_count: int = 0

    class Config:
        from_attributes = True


class JobUpdate(BaseModel):
    status: JobStatus
    error_message: Optional[str] = None


# ============ Database Dependency ============


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============ FastAPI App ============


app = FastAPI(title="Job Distribution Server")


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


# ============ SQLAdmin Setup ============


class ExperimentAdmin(ModelView, model=Experiment):
    column_list = [
        Experiment.experiment_id,
        Experiment.name,
        Experiment.status,
        "progress",
        "pending_jobs",
        "running_jobs",
        "completed_jobs",
        "failed_jobs",
        Experiment.created_at,
    ]
    column_labels = {
        "progress": "Progress",
        "pending_jobs": "Pending",
        "running_jobs": "Running",
        "completed_jobs": "Completed",
        "failed_jobs": "Failed",
    }
    column_searchable_list = [Experiment.experiment_id, Experiment.name]
    column_sortable_list = [
        Experiment.experiment_id,
        Experiment.name,
        Experiment.status,
        Experiment.created_at,
    ]
    column_default_sort = ("created_at", True)
    form_excluded_columns = [Experiment.jobs]
    name = "Experiment"
    name_plural = "Experiments"
    icon = "fa-solid fa-flask"


class JobAdmin(ModelView, model=Job):
    column_list = [
        Job.job_id,
        Job.experiment_id,
        Job.status,
        Job.worker_id,
        Job.created_at,
        Job.started_at,
        Job.completed_at,
    ]
    column_searchable_list = [Job.job_id, Job.experiment_id, Job.worker_id]
    column_sortable_list = [
        Job.job_id,
        Job.experiment_id,
        Job.status,
        Job.created_at,
        Job.started_at,
        Job.completed_at,
    ]
    column_default_sort = ("created_at", True)
    column_details_exclude_list = [Job.payload]
    name = "Job"
    name_plural = "Jobs"
    icon = "fa-solid fa-briefcase"


admin = Admin(app, engine, title="Job Server Admin")
admin.add_view(ExperimentAdmin)
admin.add_view(JobAdmin)


# ============ Helper Functions ============


def _experiment_to_response(exp: Experiment) -> ExperimentResponse:
    return ExperimentResponse(
        experiment_id=exp.experiment_id,
        name=exp.name,
        description=exp.description,
        config_file=exp.config_file,
        status=ExperimentStatus(exp.status),
        created_at=exp.created_at,
    )


def _job_to_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        experiment_id=job.experiment_id,
        payload=json.loads(job.payload),
        status=JobStatus(job.status),
        worker_id=job.worker_id,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        failure_count=job.failure_count,
    )


# ============ Experiment Endpoints ============


@app.post("/experiments", response_model=ExperimentResponse)
def create_experiment(exp: ExperimentCreate, db: Session = Depends(get_db)):
    """Create a new experiment. Optionally creates num_jobs pending jobs."""
    experiment_id = exp.experiment_id or secrets.token_hex(8)
    created_at = time.time()

    # Check if experiment already exists
    existing = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Experiment {experiment_id} already exists")

    experiment = Experiment(
        experiment_id=experiment_id,
        name=exp.name,
        description=exp.description,
        config_file=exp.config_file,
        status=ExperimentStatus.ACTIVE.value,
        created_at=created_at,
    )
    db.add(experiment)

    # Create job entries if num_jobs is specified
    if exp.num_jobs is not None and exp.num_jobs > 0:
        for i in range(exp.num_jobs):
            job = Job(
                job_id=f"{experiment_id}_job_{i}",
                experiment_id=experiment_id,
                payload=json.dumps({"job_index": i}),
                status=JobStatus.PENDING.value,
                created_at=created_at,
            )
            db.add(job)

    db.commit()
    db.refresh(experiment)
    return _experiment_to_response(experiment)


@app.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Get experiment details."""
    experiment = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return _experiment_to_response(experiment)


@app.get("/experiments")
def list_experiments(db: Session = Depends(get_db)):
    """List all experiments."""
    experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    return [_experiment_to_response(exp) for exp in experiments]


@app.get("/experiments/{experiment_id}/stats")
def get_experiment_stats(experiment_id: str, db: Session = Depends(get_db)):
    """Get job statistics for an experiment."""
    experiment = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    stats = {}
    for status in JobStatus:
        count = (
            db.query(func.count(Job.job_id))
            .filter(Job.experiment_id == experiment_id, Job.status == status.value)
            .scalar()
        )
        stats[status.value] = count

    stats["total"] = sum(stats.values())
    return stats


@app.delete("/experiments/{experiment_id}")
def delete_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Delete an experiment and all its jobs."""
    experiment = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    db.delete(experiment)
    db.commit()
    return {"deleted": experiment_id}


# ============ Job Endpoints ============


@app.post("/experiments/{experiment_id}/jobs", response_model=JobResponse)
def submit_job(experiment_id: str, job: JobSubmission, db: Session = Depends(get_db)):
    """Submit a new job to an experiment."""
    experiment = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    existing = db.query(Job).filter(Job.job_id == job.job_id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Job {job.job_id} already exists")

    new_job = Job(
        job_id=job.job_id,
        experiment_id=experiment_id,
        payload=json.dumps(job.payload),
        status=JobStatus.PENDING.value,
        created_at=time.time(),
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)
    return _job_to_response(new_job)


@app.post("/experiments/{experiment_id}/jobs/batch")
def submit_jobs_batch(experiment_id: str, jobs: list[JobSubmission], db: Session = Depends(get_db)):
    """Submit multiple jobs to an experiment."""
    experiment = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    submitted = []
    created_at = time.time()
    for job in jobs:
        existing = db.query(Job).filter(Job.job_id == job.job_id).first()
        if existing:
            continue  # Skip duplicates

        new_job = Job(
            job_id=job.job_id,
            experiment_id=experiment_id,
            payload=json.dumps(job.payload),
            status=JobStatus.PENDING.value,
            created_at=created_at,
        )
        db.add(new_job)
        submitted.append(job.job_id)

    db.commit()
    return {"submitted": len(submitted), "job_ids": submitted}


@app.get("/experiments/{experiment_id}/jobs/fetch")
def fetch_job(experiment_id: str, worker_id: str, db: Session = Depends(get_db)) -> Optional[JobResponse]:
    """
    Fetch a pending job from an experiment and mark it as running.
    Returns null if no jobs are available.
    """
    experiment = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    job = (
        db.query(Job)
        .filter(Job.experiment_id == experiment_id, Job.status == JobStatus.PENDING.value)
        .order_by(Job.created_at.asc())
        .with_for_update(skip_locked=True)
        .first()
    )

    if job is None:
        return None

    job.status = JobStatus.RUNNING.value
    job.worker_id = worker_id
    job.started_at = time.time()
    db.commit()
    db.refresh(job)
    return _job_to_response(job)


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)):
    """Get job details by ID."""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_to_response(job)


@app.put("/jobs/{job_id}")
def update_job(job_id: str, update: JobUpdate, db: Session = Depends(get_db)) -> JobResponse:
    """Update job status (complete or fail)."""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job.error_message = update.error_message

    if update.status == JobStatus.COMPLETED:
        job.status = JobStatus.COMPLETED.value
        job.completed_at = time.time()
    elif update.status == JobStatus.FAILED:
        job.failure_count += 1
        if job.failure_count >= MAX_JOB_RETRIES:
            job.status = JobStatus.FAILED.value
            job.completed_at = time.time()
        else:
            job.status = JobStatus.PENDING.value

    # Check if experiment is done (no pending or running jobs)
    if job.status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
        pending_or_running = (
            db.query(func.count(Job.job_id))
            .filter(
                Job.experiment_id == job.experiment_id,
                Job.status.in_([JobStatus.PENDING.value, JobStatus.RUNNING.value]),
            )
            .scalar()
        )
        if pending_or_running == 0:
            experiment = (
                db.query(Experiment)
                .filter(Experiment.experiment_id == job.experiment_id)
                .first()
            )
            if experiment:
                experiment.status = ExperimentStatus.COMPLETED.value

    db.commit()
    db.refresh(job)
    return _job_to_response(job)


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, db: Session = Depends(get_db)):
    """Delete a job."""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    db.delete(job)
    db.commit()
    return {"deleted": job_id}


# ============ Global Stats ============


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get global job queue statistics."""
    stats = {}
    for status in JobStatus:
        count = db.query(func.count(Job.job_id)).filter(Job.status == status.value).scalar()
        stats[status.value] = count

    stats["total"] = sum(stats.values())
    stats["experiments"] = db.query(func.count(Experiment.experiment_id)).scalar()
    return stats


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
