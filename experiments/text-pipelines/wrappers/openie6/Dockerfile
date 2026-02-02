FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/dair-iitd/openie6.git

WORKDIR /app/openie6

RUN sed -i '/^torch==/d' requirements.txt

RUN sed -i '/^absl-py==/d' requirements.txt

RUN pip install "pip<21" "setuptools<58" "wheel<0.38" && \
    pip install -r requirements.txt && \
    python -m nltk.downloader stopwords punkt

RUN zenodo_get 4054476 && tar -xvf openie6_data.tar.gz && \
    zenodo_get 4055395 && tar -xvf openie6_models.tar.gz

COPY openie6.sh /usr/local/bin/openie6.sh
RUN chmod +x /usr/local/bin/openie6.sh

