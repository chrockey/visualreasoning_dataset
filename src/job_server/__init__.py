"""
Job Distribution Server

A simple, self-hosted job queue for distributed processing.

Components:
    - server: FastAPI server with SQLite persistence
    - worker: Base class for job workers
    - client: Client for job submission and monitoring
"""
