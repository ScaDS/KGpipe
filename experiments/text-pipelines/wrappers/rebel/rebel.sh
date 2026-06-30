#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage:"
  echo "  rebel.sh <input.txt> <output.txt>"
  echo "  rebel.sh <input_folder> <output_folder>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"


python /app/rebel_cli.py "$INPUT" "$OUTPUT"

echo "All files processed."

exit 0