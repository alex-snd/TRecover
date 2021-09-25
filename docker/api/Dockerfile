# Stage 1: Builder/Compiler
FROM python:3.8-slim AS builder

# Copy necessary data
COPY ./app /workspace/app
COPY ./ml /workspace/ml
COPY ./utils /workspace/utils
COPY ./config.py /workspace/config.py
COPY ./initialize.py /workspace/initialize.py
COPY ./requirements/api.txt /requirements.txt

# Install packages
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
	pip install --no-cache-dir -r /requirements.txt

# Initialize the project
RUN python3.8 ./workspace/initialize.py
	

# Stage 2: Runtime
FROM python:3.8-slim AS runtime

ARG CUDA
ARG CELERY_BROKER
ARG CELERY_BACKEND
ARG CELERY_WORKERS
ARG FASTAPI_HOST
ARG FASTAPI_PORT
ARG FASTAPI_WORKERS

ENV CUDA ${CUDA:-"True"}
ENV CELERY_BROKER ${CELERY_BROKER:-"pyamqp://guest@localhost:5672"}
ENV CELERY_BACKEND ${CELERY_BACKEND:-"redis://localhost:6379"}
ENV FASTAPI_HOST ${FASTAPI_HOST:-"api"}
ENV FASTAPI_PORT ${FASTAPI_PORT:-8001}
ENV FASTAPI_WORKERS ${FASTAPI_WORKERS:-1}
	
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder ./workspace ./workspace

CMD uvicorn workspace.app.api.zreaderapi:api --host 0.0.0.0 --port ${FASTAPI_PORT:-8001} --workers ${FASTAPI_WORKERS:-1}

# API port
EXPOSE ${FASTAPI_PORT:-8001}