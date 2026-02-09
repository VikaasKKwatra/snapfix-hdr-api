import os
import redis
from rq import Worker, Queue

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
print(f"[run_worker] Connecting to Redis: {redis_url[:20]}...")
conn = redis.from_url(redis_url)
q = Queue("hdr", connection=conn)
worker = Worker([q], connection=conn)
print("[run_worker] Starting worker on queue 'hdr'...")
worker.work()
