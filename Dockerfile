# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (Fly.io uses PORT environment variable)
EXPOSE 8080

# Set the PORT environment variable
ENV PORT 8080

# Command to run the app with Waitress
CMD ["sh", "-c", "waitress-serve --port=${PORT} app:app"]