web: uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
worker: rq worker hdr_jobs --url ${REDIS_URL}
