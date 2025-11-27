"""
Job Client - Submit jobs and monitor status

Usage (Python):
    from src.job_server.client import JobClient

    client = JobClient("http://localhost:8000")
    exp = client.create_experiment(name="My Experiment")
    print(f"Created: {client.experiment_id}")

    client.submit("job-001", {"video_path": "/data/video1.mp4"})
    print(client.stats())

CLI:
    python -m src.job_server.client create --name "My Experiment"
    python -m src.job_server.client list
    python -m src.job_server.client get EXPERIMENT_ID
    python -m src.job_server.client delete EXPERIMENT_ID
"""

import json
import time
from typing import Optional

import requests


class JobClient:
    """Client for interacting with the job server."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        experiment_id: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.experiment_id = experiment_id
        self.timeout = timeout

    def _require_experiment(self):
        if self.experiment_id is None:
            raise ValueError(
                "experiment_id is required. Set it in constructor or use set_experiment()"
            )

    # ============ Experiment Methods ============

    def set_experiment(self, experiment_id: str):
        """Set the current experiment ID."""
        self.experiment_id = experiment_id

    def create_experiment(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        num_jobs: Optional[int] = None,
    ) -> dict:
        """Create a new experiment. Server auto-generates ID if not provided."""
        response = requests.post(
            f"{self.server_url}/experiments",
            json={
                "name": name,
                "description": description,
                "num_jobs": num_jobs,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        self.experiment_id = result["experiment_id"]
        return result

    def get_experiment(self, experiment_id: Optional[str] = None) -> dict:
        """Get experiment details."""
        exp_id = experiment_id or self.experiment_id
        if not exp_id:
            raise ValueError("experiment_id required")
        response = requests.get(
            f"{self.server_url}/experiments/{exp_id}", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def list_experiments(self) -> list:
        """List all experiments."""
        response = requests.get(f"{self.server_url}/experiments", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def pause_experiment(self, experiment_id: Optional[str] = None) -> dict:
        """Pause an experiment (no new jobs will be fetched)."""
        exp_id = experiment_id or self.experiment_id
        if not exp_id:
            raise ValueError("experiment_id required")
        response = requests.put(
            f"{self.server_url}/experiments/{exp_id}/pause", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def resume_experiment(self, experiment_id: Optional[str] = None) -> dict:
        """Resume a paused experiment."""
        exp_id = experiment_id or self.experiment_id
        if not exp_id:
            raise ValueError("experiment_id required")
        response = requests.put(
            f"{self.server_url}/experiments/{exp_id}/resume", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def delete_experiment(self, experiment_id: Optional[str] = None) -> dict:
        """Delete an experiment and all its jobs."""
        exp_id = experiment_id or self.experiment_id
        if not exp_id:
            raise ValueError("experiment_id required")
        response = requests.delete(
            f"{self.server_url}/experiments/{exp_id}", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    # ============ Job Methods ============

    def submit(self, experiment_id: str, job_id: str, payload: str | dict) -> dict:
        """Submit a single job. Payload can be JSON string or dict."""
        if isinstance(payload, str):
            payload = json.loads(payload)
        response = requests.post(
            f"{self.server_url}/experiments/{experiment_id}/jobs",
            json={"job_id": job_id, "payload": payload},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def submit_batch(
        self, jobs: list[tuple[str, dict]], chunk_size: int = 1000
    ) -> dict:
        """Submit multiple jobs to the current experiment."""
        self._require_experiment()
        all_submitted = []
        for i in range(0, len(jobs), chunk_size):
            chunk = jobs[i : i + chunk_size]
            data = [{"job_id": jid, "payload": p} for jid, p in chunk]
            response = requests.post(
                f"{self.server_url}/experiments/{self.experiment_id}/jobs/batch",
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            all_submitted.extend(result.get("job_ids", []))
        return {"submitted": len(all_submitted), "job_ids": all_submitted}

    def get_job(self, job_id: str) -> dict:
        """Get job details."""
        response = requests.get(
            f"{self.server_url}/jobs/{job_id}", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def stats(self, experiment_id: Optional[str] = None) -> dict:
        """Get statistics for an experiment."""
        exp_id = experiment_id or self.experiment_id
        if not exp_id:
            raise ValueError("experiment_id required")
        response = requests.get(
            f"{self.server_url}/experiments/{exp_id}/stats", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def global_stats(self) -> dict:
        """Get global statistics across all experiments."""
        response = requests.get(f"{self.server_url}/stats", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def retry(self, job_id: str) -> dict:
        """Retry a failed job."""
        response = requests.post(
            f"{self.server_url}/jobs/{job_id}/retry", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def retry_all_failed(self, experiment_id: Optional[str] = None) -> dict:
        """Retry all failed jobs in an experiment."""
        exp_id = experiment_id or self.experiment_id
        if not exp_id:
            raise ValueError("experiment_id required")
        response = requests.post(
            f"{self.server_url}/experiments/{exp_id}/jobs/retry-all-failed",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def delete_job(self, job_id: str) -> dict:
        """Delete a job."""
        response = requests.delete(
            f"{self.server_url}/jobs/{job_id}", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def wait(self, experiment_id: Optional[str] = None, interval: float = 10.0) -> dict:
        """Wait until all jobs in the experiment are completed or failed."""
        exp_id = experiment_id or self.experiment_id
        if not exp_id:
            raise ValueError("experiment_id required")

        print(f"Waiting for experiment {exp_id} to complete...")
        while True:
            s = self.stats(exp_id)
            total = s.get("total", 0)
            completed = s.get("completed", 0)
            failed = s.get("failed", 0)
            running = s.get("running", 0)
            pending = s.get("pending", 0)
            pct = (completed + failed) / total * 100 if total > 0 else 0
            print(
                f"\rProgress: {pct:.1f}% ({completed + failed}/{total}) | Running: {running} | Pending: {pending}   ",
                end="",
            )

            if pending == 0 and running == 0:
                print(f"\nDone! Completed: {completed}, Failed: {failed}")
                return s

            time.sleep(interval)


def _client(server: str = "http://localhost:8000") -> JobClient:
    return JobClient(server)


def create_experiment(
    name: Optional[str] = None,
    description: Optional[str] = None,
    num_jobs: Optional[int] = None,
    server: str = "http://localhost:8000",
):
    """Create a new experiment."""
    return _client(server).create_experiment(name, description, num_jobs)


def list_experiments(server: str = "http://localhost:8000"):
    """List all experiments."""
    return _client(server).list_experiments()


def get_experiment(experiment_id: str, server: str = "http://localhost:8000"):
    """Get experiment details."""
    return _client(server).get_experiment(experiment_id)


def delete_experiment(experiment_id: str, server: str = "http://localhost:8000"):
    """Delete an experiment and all its jobs."""
    return _client(server).delete_experiment(experiment_id)


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "create": create_experiment,
            "list": list_experiments,
            "get": get_experiment,
            "delete": delete_experiment,
        }
    )
