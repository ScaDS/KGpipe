FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install transformers torch

COPY bin/rebel_cli.py /app/rebel_cli.py
COPY rebel.sh /usr/local/bin/rebel.sh

RUN chmod +x /usr/local/bin/rebel.sh


