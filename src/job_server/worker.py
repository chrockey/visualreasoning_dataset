"""
Job Worker Base Class

Workers poll the server for jobs, process them, and report completion/failure.
Models are loaded once when the worker starts.

Usage:
    class MyWorker(BaseWorker):
        def setup(self):
            self.model = load_heavy_model()

        def process_job(self, job_id: str, payload: dict):
            result = self.model.predict(payload["input"])
            # Save result locally or to shared storage

    worker = MyWorker(server_url="http://localhost:8000")
    worker.run()
"""

import logging
import socket
import time
import traceback
from abc import ABC, abstractmethod
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """
    Base class for job workers.

    Subclass this and implement:
        - setup(): Load models, initialize resources
        - process_job(job_id, payload): Process a single job
    """

    def __init__(
        self,
        server_url: str,
        worker_id: Optional[str] = None,
        poll_interval: float = 5.0,
        max_retries: int = 3,
    ):
        """
        Args:
            server_url: URL of the job server (e.g., "http://localhost:8000")
            worker_id: Unique worker identifier. If None, uses hostname + timestamp.
            poll_interval: Seconds to wait between polling when no jobs available.
            max_retries: Number of retries for network errors.
        """
        self.server_url = server_url.rstrip("/")
        self.worker_id = worker_id or f"{socket.gethostname()}-{int(time.time())}"
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self._running = False

    @abstractmethod
    def setup(self):
        """
        Called once before processing starts.
        Use this to load models and initialize resources.
        """
        pass

    @abstractmethod
    def process_job(self, job_id: str, payload: dict):
        """
        Process a single job.

        Args:
            job_id: Unique job identifier
            payload: Job data (whatever was submitted)

        Raises:
            Any exception will mark the job as failed.
        """
        pass

    def teardown(self):
        """Called when worker stops. Override to clean up resources."""
        pass

    def run(self, max_jobs: Optional[int] = None):
        """
        Start the worker loop.

        Args:
            max_jobs: Stop after processing this many jobs. None = run forever.
        """
        logger.info(f"Worker {self.worker_id} starting...")
        logger.info(f"Server: {self.server_url}")

        # Setup phase
        logger.info("Running setup...")
        self.setup()
        logger.info("Setup complete. Starting job loop.")

        self._running = True
        jobs_processed = 0

        try:
            while self._running:
                if max_jobs is not None and jobs_processed >= max_jobs:
                    logger.info(f"Reached max jobs ({max_jobs}). Stopping.")
                    break

                job = self._fetch_job()

                if job is None:
                    logger.debug(f"No jobs available. Waiting {self.poll_interval}s...")
                    time.sleep(self.poll_interval)
                    continue

                job_id = job["job_id"]
                payload = job["payload"]

                logger.info(f"Processing job: {job_id}")
                start_time = time.time()

                try:
                    self.process_job(job_id, payload)
                    elapsed = time.time() - start_time
                    logger.info(f"Job {job_id} completed in {elapsed:.1f}s")
                    self._update_job(job_id, "completed")
                except Exception as e:
                    elapsed = time.time() - start_time
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(f"Job {job_id} failed after {elapsed:.1f}s: {error_msg}")
                    self._update_job(job_id, "failed", error_msg)

                jobs_processed += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Shutting down...")
        finally:
            self._running = False
            self.teardown()
            logger.info(f"Worker stopped. Processed {jobs_processed} jobs.")

    def stop(self):
        """Signal the worker to stop after current job."""
        self._running = False

    def _fetch_job(self) -> Optional[dict]:
        """Fetch a job from the server."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.server_url}/jobs/fetch",
                    params={"worker_id": self.worker_id},
                    timeout=30,
                )
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 204 or response.json() is None:
                    return None
                else:
                    logger.warning(f"Unexpected response: {response.status_code}")
                    return None
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
        return None

    def _update_job(self, job_id: str, status: str, error_message: Optional[str] = None):
        """Update job status on the server."""
        data = {"status": status}
        if error_message:
            data["error_message"] = error_message[:10000]  # Truncate long errors

        for attempt in range(self.max_retries):
            try:
                response = requests.put(
                    f"{self.server_url}/jobs/{job_id}",
                    json=data,
                    timeout=30,
                )
                if response.status_code == 200:
                    return
                else:
                    logger.warning(f"Failed to update job {job_id}: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error updating job (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

        logger.error(f"Failed to update job {job_id} after {self.max_retries} attempts")


# Example worker implementation
class ExampleWorker(BaseWorker):
    """Example worker that simulates processing."""

    def setup(self):
        logger.info("Loading models... (simulated)")
        time.sleep(2)  # Simulate model loading
        logger.info("Models loaded!")

    def process_job(self, job_id: str, payload: dict):
        # Simulate processing
        duration = payload.get("duration", 5)
        logger.info(f"Processing for {duration}s...")
        time.sleep(duration)

        # Simulate occasional failures
        if payload.get("fail", False):
            raise ValueError("Simulated failure")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run example worker")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--worker-id", help="Worker ID")
    parser.add_argument("--max-jobs", type=int, help="Max jobs to process")
    args = parser.parse_args()

    worker = ExampleWorker(
        server_url=args.server,
        worker_id=args.worker_id,
    )
    worker.run(max_jobs=args.max_jobs)
