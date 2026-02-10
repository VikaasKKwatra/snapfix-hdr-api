import redis
from rq import Worker, Queue, Connection

import os

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
conn = redis.from_url(redis_url)

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker(["hdr"])
        worker.work()
