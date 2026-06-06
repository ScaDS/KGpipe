.PHONY:

DATASET_URL := https://zenodo.org/record/17246358/files/inc_movie_kg_datasets.tar.gz?download=1
BASE_DIR := ./data

# === Main ===

setup_docker: download-datasets kgpipe_docker kgtools start-services

run_docker_small:
	bash ../../scripts/moviekg_docker.sh $(BASE_DIR) docker.env

pipelines:
	pytest -v src/moviekg/pipelines/ -k "not llm"

pipelines-llm:
	pytest -v src/moviekg/pipelines/ -k "llm"

# === Docker ===

$(BASE_DIR)/.kgpipe-docker-built:
	@cd ../.. && make docker_build
	touch $@

kgpipe_docker: $(BASE_DIR)/.kgpipe-docker-built

$(BASE_DIR)/kgtools/README.md:
	@cd $(BASE_DIR) && git clone https://github.com/vehnem/kg-tools.git kgtools

$(BASE_DIR)/kgtools/.paris-docker-built	:
	@cd $(BASE_DIR)/kgtools/paris && make docker_build
	touch $@

$(BASE_DIR)/kgtools/.valentine-docker-built:
	@cd $(BASE_DIR)/kgtools/valentine && make docker_build
	touch $@

$(BASE_DIR)/kgtools/.pyjedai-docker-built:
	@cd $(BASE_DIR)/kgtools/pyjedai && make docker_build
	touch $@

$(BASE_DIR)/kgtools/.corenlp-docker-built:
	@cd $(BASE_DIR)/kgtools/corenlp && make docker_build
	touch $@

$(BASE_DIR)/kgtools/.dbpedia-spotlight-docker-built:
	@cd $(BASE_DIR)/kgtools/dbpedia-spotlight && make docker_build
	touch $@

$(BASE_DIR)/kgtools/.docker-built: $(BASE_DIR)/kgtools/.paris-docker-built $(BASE_DIR)/kgtools/.valentine-docker-built $(BASE_DIR)/kgtools/.pyjedai-docker-built $(BASE_DIR)/kgtools/.corenlp-docker-built $(BASE_DIR)/kgtools/.dbpedia-spotlight-docker-built
	touch $@

kgtools_clone: $(BASE_DIR)/kgtools/README.md

kgtools_docker: $(BASE_DIR)/kgtools/.docker-built

kgtools: kgtools_clone kgtools_docker

# === Setup ===

start-services:
	@cd ../../ && docker compose up -d

clean:
	rm -rf data;

# === Dataset ===

$(BASE_DIR)/datasets.tar.gz:
	@mkdir -p $(BASE_DIR)
	@cd $(BASE_DIR) && wget $(DATASET_URL) -O datasets.tar.gz

$(BASE_DIR)/datasets/.extracted: $(BASE_DIR)/datasets.tar.gz
	@mkdir -p $(BASE_DIR)/datasets
	@cd $(BASE_DIR) && tar -xzf datasets.tar.gz
	@touch $(BASE_DIR)/datasets/.extracted

download-datasets: $(BASE_DIR)/datasets/.extracted

datasets-eval:
	pytest -v src/moviekg/datasets/test_evaluate_film_data.py

# === RDF ===

test-rdf-base:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_base

test-rdf-alt:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_alt

test-rdf-llm:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_llm

# === JSON ===

test-json-base:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k json_base

test-json-alt:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k json_alt

test-json-llm:
	pytest -v -s src/moviekg/pipelines/test_inc_ssp.py -k json_llm

# === TEXT ===

test-text-base:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k text_base

test-text-alt:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k text_alt

test-text-llm:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k text_llm

# === MSPs ===

test-msp-all:
	pytest -s -v src/moviekg/pipelines/test_inc_msp.py
