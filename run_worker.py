"""Entry point for the Railway worker service."""
import os
from redis import Redis
from rq import Worker, Queue

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

if __name__ == "__main__":
    redis_conn = Redis.from_url(REDIS_URL)
    queue = Queue("hdr_jobs", connection=redis_conn)
    worker = Worker([queue], connection=redis_conn)
    print(f"[Worker] Starting RQ worker, listening on queue 'hdr_jobs'")
    print(f"[Worker] Redis: {REDIS_URL[:30]}...")
    worker.work(with_scheduler=False)
