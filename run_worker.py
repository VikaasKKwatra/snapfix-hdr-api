import os
import redis
from rq import Worker, Queue

def main():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    conn = redis.from_url(redis_url)
    queues = [Queue("hdr_jobs", connection=conn)]
    worker = Worker(queues, connection=conn)
    print(f"[Worker] Starting, listening on queue 'hdr_jobs'")
    worker.work(with_scheduler=False)

if __name__ == "__main__":
    main()
