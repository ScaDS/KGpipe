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

evaluation:
	pytest -v src/moviekg/evaluation/ -k "not llm"

paper:
	pytest -v src/moviekg/evaluation/test_inc_msp_evaluation.py -k concat;
	pytest -v src/moviekg/paper/test_figtab.py;

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

# === SSPs ===

test-ssp-all:
	time pytest -s -v src/moviekg/pipelines/test_inc_ssp.py

eval-ssp-all:
	time pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py

test-ssp-classic:
	time pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k "not llm_"

test-ssp-llm:
	time pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k "llm_"

eval-ssp-llm:
	time pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k "llm_"

# === RDF ===

test-rdf-a:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_a

eval-rdf-a:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k rdf_a

test-rdf-b:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_b

eval-rdf-b:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k rdf_b

test-rdf-c:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_llm

eval-rdf-c:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k rdf_llm

# === JSON ===

test-json-a:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k json_a

eval-json-a:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k json_a

test-json-b:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k json_b

eval-json-b:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k json_b

test-json-c:
	pytest -v -s src/moviekg/pipelines/test_inc_ssp.py -k json_llm

eval-json-c:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k json_llm

# === TEXT ===

test-text-a:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k text_a

eval-text-a:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k text_a

test-text-b:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k text_b

eval-text-b:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k text_b

test-text-c:
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k text_llm

eval-text-c:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k text_llm

# === MSPs ===

test-msp-all:
	pytest -s -v src/moviekg/pipelines/test_inc_msp.py

eval-msp-all:
	pytest -s -v src/moviekg/evaluation/test_inc_msp_evaluation.py

eval-msp-rjt:
	pytest -s -v src/moviekg/evaluation/test_inc_msp_evaluation.py -k rdf-json-text

# === Paper ===

concatenate-metrics:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k test_concatenate_long_table_rows

paper-figtab:
	pytest -s -v src/moviekg/paper/test_figtab.py
