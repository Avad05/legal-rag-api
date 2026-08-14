FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and corpus
COPY src/ ./src/
COPY corpus/ ./corpus/
COPY .env.example .env

# Expose FastAPI default port
EXPOSE 8000

# Start server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
