FROM python:3.6-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/dair-iitd/imojie.git
WORKDIR /app/imojie

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN zenodo_get 3779954 && zenodo_get 3780045 && zenodo_get 3775983

RUN for f in *.tar.gz; do tar -xvzf "$f"; done

RUN python -m nltk.downloader stopwords punkt

COPY imojie.sh /usr/local/bin/imojie.sh
RUN chmod +x /usr/local/bin/imojie.sh

