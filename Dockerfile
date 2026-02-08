FROM python:3.12-slim

# system deps for opencv + image processing (safe basics)
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements first to cache installs
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the code
COPY . /app

# railway uses PORT for web service


# default command (web) – worker will override this in Railway start command
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
