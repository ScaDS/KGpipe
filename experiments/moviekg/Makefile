# TODO

.PHONY: create-repos update-repos create-env install-deps download-data test reproduce-tests-only

EVN_NAME := moviekg
DATA_URL := https://cloud.scadsai.uni-leipzig.de/index.php/s/no5jySLKBRXXqx6/download

create-repos:
	@test -d git.kgflex || git clone git@github.com:scads/kgflex.git --branch paper2025 git.kgflex;
	@test -d git.odibel || 	git clone git@github.com:scads/odibel.git --branch paper2025 git.odibel;
	@test -d git.kgtools || git clone git@github.com:vehnem/kg-tools.git --branch paper2025 git.kgtools;

update-repos:
	cd git.kgflex && git pull origin paper2025;
	cd git.odibel && git pull origin paper2025;
	cd git.kgtools && git pull origin paper2025;

create-env:
	@micromamba info --envs | grep -q "^$(ENV_NAME) " || \
	micromamba create -n $(EVN_NAME) python=3.12 -c conda-forge;

install-deps:
	cd git.kgflex && micromamba run -n $(EVN_NAME) pip install -e .; 
	cd git.odibel && micromamba run -n $(EVN_NAME) pip install -e .; 

#install-kgtools:
# TODO: build tools in docker
# for now push directly to registry
#	cd git.kgtools && bash build.sh; 

data/data.tar.bz2:
	@mkdir -p data
	@cd data && wget $(DATA_URL) -O data.tar.bz2 && tar -xjf data.tar.bz2

download-data: data/data.tar.bz2#

test:
	echo "Executing all tests...";
	micromamba run -n $(EVN_NAME) pytest -v src/moviekg/;

reproduce-tests-only: create-repos update-repos create-env download-data test

clean:
	rm -rf data;

# === Services ===

start-services:
# docker run -d --name spotlight -p 2222:2222 -p 8080:8080 -p 6100:6100 -p 6101:6101 -p 6102:610
# docker run falcon
# docker run ollama
	echo TODO

# === Dataset ===

eval-dataset:
	pytest -s -v src/moviekg/datasets/test_evaluate_film_data.py

# === SSPs ===

test-ssp-all:
	time pytest -s -v src/moviekg/pipelines/test_inc_ssp.py

eval-ssp-all:
	time pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py

test-ssp-classic:
	time pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k "not llm_"

#eval-ssp-classic:
#	time pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k "not gpt"

test-ssp-llm:
	time pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k "llm_"

eval-ssp-llm:
	time pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k "llm_"

# === RDF ===

test-rdf-a:
# sudo rm -rf out/rdf_a;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_a

eval-rdf-a:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k rdf_a

test-rdf-b:
# sudo rm -rf out/rdf_b;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_b

eval-rdf-b:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k rdf_b


test-rdf-b2:
# sudo rm -rf out/rdf_b;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_b2

eval-rdf-b2:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k rdf_b2


test-rdf-c:
# sudo rm -rf out/rdf_b;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_llm

test-rdf-llm-schema-align:
# sudo rm -rf out/rdf_llm_schema_align;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_llm_schema_align

eval-rdf-llm-schema-align:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k rdf_llm_schema_align

test-rdf-llm-full-integration:
# sudo rm -rf out/rdf_llm_full_integration;
	pytest -v src/moviekg/pipelines/test_inc_ssp.py -k rdf_llm_full_integration

eval-rdf-llm-full-integration:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k rdf_llm_full_integration

# === JSON ===

test-json-a:
# sudo rm -rf out/json_a;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k json_a

eval-json-a:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k json_a

test-json-b:
# sudo rm -rf out/json_b;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k json_b	

eval-json-b:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k json_b

test-json-b2:
# sudo rm -rf out/json_b;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k json_b2

test-json-llm-schema-align:
# sudo rm -rf out/json_llm_schema_align;
	pytest -v -s src/moviekg/pipelines/test_inc_ssp.py -k json_llm

eval-json-llm-schema-align:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k json_llm

test-json-llm-semi-integration:
# sudo rm -rf out/json_llm_semi_integration;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k json_llm_semi_integration

eval-json-llm-semi-integration:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k json_llm_semi_integration

test-json-llm-full-integration:
# sudo rm -rf out/json_llm_full_integration;
	pytest -v src/moviekg/pipelines/test_inc_ssp.py -k json_llm_full_integration

eval-json-llm-full-integration:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k json_llm_full_integration

# === TEXT ===

test-text-a:
# sudo rm -rf out/text_a;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k text_a

eval-text-a:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k text_a

test-text-b:
# sudo rm -rf out/text_b;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k text_b

eval-text-b:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k text_b

test-text-llm-semi-integration:
# sudo rm -rf out/text_llm_semi_integration;
	pytest -s -v src/moviekg/pipelines/test_inc_ssp.py -k text_llm

eval-text-llm-semi-integration:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k text_llm

test-text-llm-full-integration:
# sudo rm -rf out/text_llm_full_integration;
	pytest -s-v src/moviekg/pipelines/test_inc_ssp.py -k text_llm_full_integration

eval-text-llm-full-integration:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k text_llm_full_integration

# === MSPs ===

test-msp-all:
	pytest -s -v src/moviekg/pipelines/test_inc_msp.py

eval-msp-all:
	pytest -s -v src/moviekg/evaluation/test_inc_msp_evaluation.py

# === Paper ===

concatenate-metrics:
	pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k test_concatenate_long_table_rows
# micromamba run -n $(EVN_NAME) pytest -s -v src/moviekg/evaluation/test_inc_ssp_evaluation.py -k test_replace_with_dict

paper-figtab:
	pytest -s -v src/moviekg/paper/test_figtab.py
