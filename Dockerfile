FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --prefer-binary -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]