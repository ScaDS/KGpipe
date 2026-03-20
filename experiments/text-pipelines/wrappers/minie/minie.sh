#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage:"
  echo "  minie.sh <input_file> <output_file>"
  echo "  minie.sh <input_folder> <output_folder>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ -f "$INPUT" ]; then
  if [ -d "$OUTPUT" ]; then
    echo "Error: Output must be a file when input is a file"
    exit 1
  fi

  java -cp "/app/minie/target/classes:/app/minie/target/dependency/*" \
       MinIEApp "$INPUT" "$OUTPUT"
  exit 0
fi

if [ -d "$INPUT" ]; then
  mkdir -p "$OUTPUT"

  for file in "$INPUT"/*; do
    if [ -f "$file" ]; then
      filename="${file##*/}"
      java -cp "/app/minie/target/classes:/app/minie/target/dependency/*" \
           MinIEApp "$file" "$OUTPUT/$filename"
    fi
  done
  exit 0
fi

echo "Error: Input must be a file or directory"
exit 1