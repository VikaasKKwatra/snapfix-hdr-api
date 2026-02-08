import os
import redis
from rq import Worker, Queue

def main():
    redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_PRIVATE_URL") or os.getenv("REDIS_PUBLIC_URL")
    if not redis_url:
        raise RuntimeError("Missing REDIS_URL (or REDIS_PRIVATE_URL / REDIS_PUBLIC_URL)")

    conn = redis.from_url(redis_url)

    queue_name = os.getenv("RQ_QUEUE", "default")
    q = Queue(queue_name, connection=conn)

    worker = Worker([q], connection=conn)
    worker.work()

if __name__ == "__main__":
    main()
