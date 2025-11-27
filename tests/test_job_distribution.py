"""
Test job distribution across multiple parallel workers.

Usage:
    cd /root/code/visualreasoning_dataset
    PYTHONPATH=. python tests/test_job_distribution.py
"""

import multiprocessing
import subprocess
import sys
import time

import requests

SERVER_URL = "http://127.0.0.1:18765"
EXPERIMENT_ID = "test-exp-001"
NUM_JOBS = 10
NUM_WORKERS = 3
JOB_DURATION = 2  # seconds per job


def start_server():
    """Start the job server."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.job_server.server:app", "--host", "127.0.0.1", "--port", "18765"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)  # Wait for server to start
    return proc


def create_experiment():
    """Create test experiment."""
    response = requests.post(
        f"{SERVER_URL}/experiments",
        json={"experiment_id": EXPERIMENT_ID, "name": "Test Experiment"},
    )
    response.raise_for_status()
    print(f"Created experiment: {EXPERIMENT_ID}")


def submit_jobs(num_jobs: int):
    """Submit test jobs."""
    jobs = [{"job_id": f"test-job-{i:03d}", "payload": {"duration": JOB_DURATION, "index": i}} for i in range(num_jobs)]
    response = requests.post(
        f"{SERVER_URL}/experiments/{EXPERIMENT_ID}/jobs/batch",
        json=jobs,
    )
    response.raise_for_status()
    result = response.json()
    print(f"Submitted {result['submitted']} jobs")
    return result


def run_worker(worker_id: str, results_queue):
    """Run a test worker that simulates processing."""
    from src.job_server.worker import BaseWorker

    class TestWorker(BaseWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.processed = []

        def setup(self):
            pass

        def process_job(self, job_id: str, payload: dict):
            print(f"[{self.worker_id}] Processing {job_id}")
            time.sleep(payload.get("duration", 1))
            self.processed.append(job_id)

        def teardown(self):
            results_queue.put((self.worker_id, self.processed))

    worker = TestWorker(
        server_url=SERVER_URL,
        experiment_id=EXPERIMENT_ID,
        worker_id=worker_id,
        poll_interval=0.5,
        stop_when_empty=True,
    )
    worker.run()


def main():
    print("=" * 60)
    print("Testing Job Distribution with Experiments")
    print("=" * 60)
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Jobs: {NUM_JOBS}, Workers: {NUM_WORKERS}, Duration: {JOB_DURATION}s/job")
    print()

    # Start server
    print("[1] Starting server...")
    server_proc = start_server()

    try:
        # Check server is running
        response = requests.get(f"{SERVER_URL}/stats")
        if response.status_code != 200:
            print("Server failed to start!")
            return
        print(f"    Server running at {SERVER_URL}")

        # Create experiment
        print(f"\n[2] Creating experiment: {EXPERIMENT_ID}")
        create_experiment()

        # Submit jobs
        print(f"\n[3] Submitting {NUM_JOBS} jobs...")
        submit_jobs(NUM_JOBS)

        # Check stats
        stats = requests.get(f"{SERVER_URL}/experiments/{EXPERIMENT_ID}/stats").json()
        print(f"    Queue: {stats}")

        # Start workers in parallel
        print(f"\n[4] Starting {NUM_WORKERS} workers in parallel...")
        results_queue = multiprocessing.Queue()

        workers = []
        for i in range(NUM_WORKERS):
            p = multiprocessing.Process(target=run_worker, args=(f"worker-{i}", results_queue))
            workers.append(p)
            p.start()

        # Wait for all workers to complete
        print("\n[5] Waiting for workers to complete...")
        for p in workers:
            p.join()

        # Collect results
        results = {}
        while not results_queue.empty():
            worker_id, processed = results_queue.get()
            results[worker_id] = processed

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        total_processed = 0
        for worker_id, jobs in sorted(results.items()):
            print(f"{worker_id}: processed {len(jobs)} jobs - {jobs}")
            total_processed += len(jobs)

        # Final stats
        stats = requests.get(f"{SERVER_URL}/experiments/{EXPERIMENT_ID}/stats").json()
        print(f"\nFinal queue stats: {stats}")
        print(f"\nTotal jobs processed: {total_processed}/{NUM_JOBS}")

        if stats["completed"] == NUM_JOBS and total_processed == NUM_JOBS:
            print("\n✅ SUCCESS: All jobs distributed and completed!")
        else:
            print("\n❌ FAILURE: Not all jobs were processed")

    finally:
        # Stop server
        print("\n[6] Stopping server...")
        server_proc.terminate()
        server_proc.wait()
        print("    Done!")


if __name__ == "__main__":
    main()
