FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    joblib \
    pandas \
    scikit-learn

RUN mkdir -p /opt/ml/model /opt/ml/input/data/train /opt/program
WORKDIR /opt/program

COPY train.py app.py serve entrypoint.sh ./

RUN chmod +x serve entrypoint.sh

ENV PYTHONUNBUFFERED=TRUE

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]
