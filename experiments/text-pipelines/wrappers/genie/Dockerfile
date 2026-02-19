FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip\
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/epfl-dlab/GenIE.git

WORKDIR /app/GenIE

RUN pip install --upgrade pip

RUN pip install -r pip_requirements.txt

RUN mkdir -p data/models

# Models initialized with a pretrained language model (GenIE - PLM) Trained on Rebel
RUN wget https://zenodo.org/record/6139236/files/genie_plm_r.ckpt \
    -O data/models/genie_plm_r.ckpt

RUN wget https://zenodo.org/record/6139236/files/tries.zip \
    && unzip tries.zip -d data && rm tries.zip

COPY bin/genie_cli.py /app/GenIE/genie_cli.py
COPY genie.sh /usr/local/bin/genie.sh
RUN chmod +x /usr/local/bin/genie.sh


