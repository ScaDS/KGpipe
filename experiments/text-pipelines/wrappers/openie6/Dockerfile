FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y \
    git wget bzip2 ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py36_4.12.0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

WORKDIR /app

RUN git clone https://github.com/dair-iitd/openie6.git

SHELL ["bash", "-c"]

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda create -n openie6 python=3.6 -y && \
    conda activate openie6 && \
    pip install --upgrade pip && \
    pip install -r openie6/requirements.txt && \
    python -m nltk.downloader stopwords punkt

WORKDIR /app/openie6
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate openie6 && \
    python zenodo_get.py 4054476 && tar -xvf openie6_data.tar.gz && \
    python zenodo_get.py 4055395 && tar -xvf openie6_models.tar.gz

COPY openie6.sh /usr/local/bin/openie6.sh
RUN chmod +x /usr/local/bin/openie6.sh

WORKDIR /app
ENTRYPOINT ["openie6.sh"]
