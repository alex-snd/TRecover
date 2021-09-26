# Stage 1: Builder/Compiler
FROM python:3.8-slim AS builder

# Copy necessary data
COPY ./app /workspace/app
COPY ./ml /workspace/ml
COPY ./utils /workspace/utils
COPY ./config.py /workspace/config.py
COPY ./initialize.py /workspace/initialize.py
COPY ./inference /workspace/inference
COPY ./requirements/all.txt /requirements.txt

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
ARG STREAMLIT_PORT
ARG FASTAPI_HOST
ARG FASTAPI_PORT
ARG FASTAPI_WORKERS

ENV CUDA ${CUDA:-"True"}
ENV CELERY_BROKER ${CELERY_BROKER:-"pyamqp://guest@localhost:5672"}
ENV CELERY_BACKEND ${CELERY_BACKEND:-"redis://localhost:6379"}
ENV STREAMLIT_PORT ${STREAMLIT_PORT:-8000}
ENV FASTAPI_HOST ${FASTAPI_HOST:-"0.0.0.0"}
ENV FASTAPI_PORT ${FASTAPI_PORT:-8001}
	
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder ./workspace ./workspace

CMD celery -A workspace.app.api.backend.celeryapp worker -l INFO -c ${CELERY_WORKERS:-1} & \
    uvicorn workspace.app.api.zreaderapi:api --host ${FASTAPI_HOST:-"0.0.0.0"} --port ${FASTAPI_PORT:-8001} --workers ${FASTAPI_WORKERS:-1} & \
    streamlit run ./workspace/app/api/dashboard.py --server.address "0.0.0.0" --server.port ${STREAMLIT_PORT:-8000} --theme.backgroundColor "#E7EAD9" --theme.secondaryBackgroundColor "#DFE3D0" --theme.primaryColor "#FF8068" --theme.textColor "#157D96" & \
    wait