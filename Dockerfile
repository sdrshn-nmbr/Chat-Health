FROM python:3.9-slim

# set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy rest of app code
COPY . .

# expose the port (fly.io uses PORT env vars)
EXPOSE 8080

# set the PORT env var
ENV PORT 8080

# command to run app with gunicorn (wsgi server)
CMD ["sh", "-c", "gunicorn app:app --workers=2 --threads=4 --bind=0.0.0.0:${PORT}"]
