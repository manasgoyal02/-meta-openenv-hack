FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV WORKERS=4
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MAX_CONCURRENT_ENVS=100

CMD ["sh", "-c", "uvicorn agriops_env.server.app:app --host $HOST --port $PORT --workers $WORKERS"]

