FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml ./
COPY vertex_proxy ./vertex_proxy
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "vertex_proxy.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
