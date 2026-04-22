"""
RQ worker process (V2A-02+ / V2A-06). Run: ``python -m api.worker`` or
``python -m rq worker default`` from repo root (PYTHONPATH must include the repo).
"""

from __future__ import annotations


def main() -> None:
    from rq import Queue, Worker

    from api.deps.redis_client import get_redis

    r = get_redis()
    queues = [Queue("default", connection=r)]
    worker = Worker(queues, connection=r, name="api-worker-1")
    worker.work()


if __name__ == "__main__":
    main()
