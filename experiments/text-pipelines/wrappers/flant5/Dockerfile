FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install transformers torch

COPY bin/flant5_cli.py /app/flant5_cli.py
COPY flant5.sh /usr/local/bin/flant5.sh

RUN chmod +x /usr/local/bin/flant5.sh


