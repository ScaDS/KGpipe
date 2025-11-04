#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-}"
ENV_FILE="${2:-}"

# Check BASE_DIR
if [[ -z "$BASE_DIR" ]]; then
    echo "Error: BASE_DIR not provided."
    echo "Usage: $0 <base_dir> <env_file>"
    exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: '$BASE_DIR' is not a directory."
    exit 1
fi

# Check ENV_FILE
if [[ -z "$ENV_FILE" ]]; then
    echo "Error: ENV_FILE not provided."
    echo "Usage: $0 <base_dir> <env_file>"
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: ENV file '$ENV_FILE' does not exist."
    exit 1
fi

echo "BASE_DIR: $BASE_DIR with realpath $(realpath $BASE_DIR)"
echo "ENV_FILE: $ENV_FILE with realpath $(realpath $ENV_FILE)"
echo "All checks passed."

OUTPUT_DIR=$(realpath $BASE_DIR)/output
DATASET_SMALL=$(realpath $BASE_DIR)/datasets/film_100

docker run --rm \
	--network=kgpipe \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-v $(realpath $BASE_DIR):$(realpath $BASE_DIR) \
	-v $(realpath $ENV_FILE):/app/experiments/moviekg/.env:ro \
    -e DATASET_SMALL=$DATASET_SMALL \
    -e OUTPUT_DIR=$OUTPUT_DIR \
	--entrypoint "bash" \
	kgpipe -c "cd /app/experiments/moviekg/; \
    make BASE_DIR=$(realpath $BASE_DIR) datasets-eval; \
    make BASE_DIR=$(realpath $BASE_DIR) pipelines; \
    make BASE_DIR=$(realpath $BASE_DIR) evaluation; \
    make BASE_DIR=$(realpath $BASE_DIR) paper"
