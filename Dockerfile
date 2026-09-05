FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml .
COPY tridifleet ./tridifleet
COPY web ./web
COPY main.py .

RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
