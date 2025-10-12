FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dongpa_engine.py app_dongpa.py ./

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app_dongpa.py", "--server.address=0.0.0.0", "--server.port=8501"]

