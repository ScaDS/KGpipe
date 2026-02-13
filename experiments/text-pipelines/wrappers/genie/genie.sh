#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: genie.sh <input.txt> <output.txt>"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

GRAPHENE_DIR="/app/GenIE"

if [ ! -f "$INPUT_FILE" ]; then
  echo "Input file not found: $INPUT_FILE"
  exit 1
fi

python /app/GenIE/genie_cli.py "$INPUT_FILE" "$OUTPUT_FILE"