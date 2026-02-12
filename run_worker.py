import os
from redis import Redis
from rq import Worker, Queue

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
conn = Redis.from_url(REDIS_URL)
queue = Queue("hdr_jobs", connection=conn)

if __name__ == "__main__":
    worker = Worker([queue], connection=conn)
    worker.work()
