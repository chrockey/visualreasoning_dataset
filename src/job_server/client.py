"""
Job Client - Submit jobs and monitor status

Usage:
    from job_server.client import JobClient

    client = JobClient("http://localhost:8000")

    # Submit single job
    client.submit("job-001", {"video_path": "/data/video1.mp4"})

    # Submit batch
    jobs = [("job-002", {"video_path": f"/data/video{i}.mp4"}) for i in range(100)]
    client.submit_batch(jobs)

    # Check status
    stats = client.stats()
    print(f"Pending: {stats['pending']}, Completed: {stats['completed']}")

CLI:
    python -m job_server.client submit --server http://localhost:8000 job-001 '{"key": "value"}'
    python -m job_server.client stats --server http://localhost:8000
"""

import json
from typing import Optional

import requests


class JobClient:
    """Client for interacting with the job server."""

    def __init__(self, server_url: str, timeout: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def submit(self, job_id: str, payload: dict) -> dict:
        """Submit a single job."""
        response = requests.post(
            f"{self.server_url}/jobs",
            json={"job_id": job_id, "payload": payload},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def submit_batch(self, jobs: list[tuple[str, dict]], chunk_size: int = 1000) -> dict:
        """
        Submit multiple jobs.

        Args:
            jobs: List of (job_id, payload) tuples
            chunk_size: Number of jobs per request

        Returns:
            {"submitted": count, "job_ids": [...]}
        """
        all_submitted = []

        for i in range(0, len(jobs), chunk_size):
            chunk = jobs[i : i + chunk_size]
            data = [{"job_id": jid, "payload": p} for jid, p in chunk]

            response = requests.post(
                f"{self.server_url}/jobs/batch",
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            all_submitted.extend(result.get("job_ids", []))

        return {"submitted": len(all_submitted), "job_ids": all_submitted}

    def get(self, job_id: str) -> dict:
        """Get job details."""
        response = requests.get(
            f"{self.server_url}/jobs/{job_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def stats(self) -> dict:
        """Get queue statistics."""
        response = requests.get(
            f"{self.server_url}/stats",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def retry(self, job_id: str) -> dict:
        """Retry a failed job."""
        response = requests.post(
            f"{self.server_url}/jobs/{job_id}/retry",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def retry_all_failed(self) -> dict:
        """Retry all failed jobs."""
        response = requests.post(
            f"{self.server_url}/jobs/retry-all-failed",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def delete(self, job_id: str) -> dict:
        """Delete a job."""
        response = requests.delete(
            f"{self.server_url}/jobs/{job_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_completion(
        self,
        poll_interval: float = 10.0,
        callback: Optional[callable] = None,
    ) -> dict:
        """
        Wait until all jobs are completed or failed.

        Args:
            poll_interval: Seconds between status checks
            callback: Optional function called with stats on each poll

        Returns:
            Final stats
        """
        import time

        while True:
            stats = self.stats()
            if callback:
                callback(stats)

            pending = stats.get("pending", 0)
            running = stats.get("running", 0)

            if pending == 0 and running == 0:
                return stats

            time.sleep(poll_interval)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Job client CLI")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a job")
    submit_parser.add_argument("job_id", help="Job ID")
    submit_parser.add_argument("payload", help="JSON payload")

    # Stats command
    subparsers.add_parser("stats", help="Get queue statistics")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get job details")
    get_parser.add_argument("job_id", help="Job ID")

    # Retry command
    retry_parser = subparsers.add_parser("retry", help="Retry a failed job")
    retry_parser.add_argument("job_id", nargs="?", help="Job ID (omit for all failed)")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a job")
    delete_parser.add_argument("job_id", help="Job ID")

    # Wait command
    wait_parser = subparsers.add_parser("wait", help="Wait for all jobs to complete")
    wait_parser.add_argument("--interval", type=float, default=10.0, help="Poll interval")

    args = parser.parse_args()
    client = JobClient(args.server)

    if args.command == "submit":
        result = client.submit(args.job_id, json.loads(args.payload))
        print(json.dumps(result, indent=2))

    elif args.command == "stats":
        stats = client.stats()
        print(f"Pending:   {stats.get('pending', 0)}")
        print(f"Running:   {stats.get('running', 0)}")
        print(f"Completed: {stats.get('completed', 0)}")
        print(f"Failed:    {stats.get('failed', 0)}")
        print(f"Total:     {stats.get('total', 0)}")

    elif args.command == "get":
        result = client.get(args.job_id)
        print(json.dumps(result, indent=2))

    elif args.command == "retry":
        if args.job_id:
            result = client.retry(args.job_id)
        else:
            result = client.retry_all_failed()
        print(json.dumps(result, indent=2))

    elif args.command == "delete":
        result = client.delete(args.job_id)
        print(json.dumps(result, indent=2))

    elif args.command == "wait":

        def print_progress(stats):
            total = stats.get("total", 0)
            completed = stats.get("completed", 0)
            failed = stats.get("failed", 0)
            running = stats.get("running", 0)
            pending = stats.get("pending", 0)
            pct = (completed + failed) / total * 100 if total > 0 else 0
            print(f"\rProgress: {pct:.1f}% ({completed + failed}/{total}) | Running: {running} | Pending: {pending}   ", end="")

        print("Waiting for jobs to complete...")
        final = client.wait_for_completion(poll_interval=args.interval, callback=print_progress)
        print(f"\nDone! Completed: {final.get('completed', 0)}, Failed: {final.get('failed', 0)}")


if __name__ == "__main__":
    main()
