FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/dair-iitd/openie6.git

WORKDIR /app/openie6

RUN conda create -n openie6 python=3.8 -y

RUN sed -i '/^torch==/d' requirements.txt

RUN sed -i '/^absl-py==/d' requirements.txt

RUN /opt/conda/envs/openie6/bin/pip install "pip<21" "setuptools<58" "wheel<0.38" "jinja2<3.1" "itsdangerous<2.0" "greenlet==1.1.3" && \
    conda run -n openie6 pip install -r requirements.txt && \
    conda run -n openie6 python -m nltk.downloader stopwords punkt

RUN conda run -n openie6 zenodo_get 4054476 && \
	tar -xvf openie6_data.tar.gz && \
	rm openie6_data.tar.gz && \
	conda run -n openie6 zenodo_get 4055395 && \
	tar -xvf openie6_models.tar.gz && \
	rm openie6_models.tar.gz


COPY openie6.sh /usr/local/bin/openie6.sh
RUN chmod +x /usr/local/bin/openie6.sh

