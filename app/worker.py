import os
import redis
from rq import Queue
from rq.worker import Worker

def main():
    redis_url = os.environ["REDIS_URL"]  # must exist in Railway variables
    conn = redis.from_url(redis_url)

    queue = Queue("default", connection=conn)
    worker = Worker([queue], connection=conn)

    worker.work()

if __name__ == "__main__":
    main()
