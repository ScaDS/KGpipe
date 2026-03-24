#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage:"
  echo "  imojie.sh <input.txt> <output_file>"
  echo "  imojie.sh <input_folder> <output_folder>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

cd /app/imojie

if [ -f "$INPUT" ]; then
  if [ -d "$OUTPUT" ]; then
    echo "Error: Output must be a file when input is a file"
    exit 1
  fi

  python standalone.py --inp "$INPUT" --out "$OUTPUT"
  exit 0
fi

if [ -d "$INPUT" ]; then
  mkdir -p "$OUTPUT"
  chmod 777 "$OUTPUT"

  for file in "$INPUT"/*; do
    if [ -f "$file" ]; then
      filename="${file##*/}"
      python standalone.py --inp "$file" --out "$OUTPUT/$filename"
    fi
  done
  exit 0
fi

echo "Error: Input must be a file or directory"
exit 1