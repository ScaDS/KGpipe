#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: imojie.sh <input.txt> <output_file>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -f "$INPUT" ]; then
  echo "Input file not found: $INPUT"
  exit 1
fi

cd /app/imojie

python standalone.py --inp $INPUT --out $OUTPUT