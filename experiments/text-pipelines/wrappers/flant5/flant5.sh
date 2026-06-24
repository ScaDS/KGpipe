#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage:"
  echo "  flant5.sh <input.txt> <output.txt>"
  echo "  flant5.sh <input_folder> <output_folder>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"


python /app/flant5_cli.py "$INPUT" "$OUTPUT"

echo "All files processed."

exit 0