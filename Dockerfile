FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY pages ./pages

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "backtest.py", "--server.address=0.0.0.0", "--server.port=8501"]
