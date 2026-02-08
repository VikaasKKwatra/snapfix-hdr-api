import os
import redis
from rq import Worker, Queue, Connection

def main():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL is missing")

    conn = redis.from_url(redis_url)

    queue_name = os.getenv("QUEUE_NAME", "hdr")

    with Connection(conn):
        worker = Worker([Queue(queue_name)])
        worker.work()

if __name__ == "__main__":
    main()
