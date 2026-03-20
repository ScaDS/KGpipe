#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage:"
  echo "  genie.sh <input.txt> <output.txt>"
  echo "  genie.sh <input_folder> <output_folder>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

GRAPHENE_DIR="/app/GenIE"

if [ -f "$INPUT" ]; then
  if [ -d "$OUTPUT" ]; then
    echo "Error: Output must be a file when input is a file"
    exit 1
  fi

  echo "Processing single file..."
  python /app/GenIE/genie_cli.py "$INPUT" "$OUTPUT"
  echo "Done."
  exit 0
fi

if [ -d "$INPUT" ]; then
  if [ -f "$OUTPUT" ]; then
    echo "Error: Output must be a folder when input is a folder"
    exit 1
  fi
  mkdir -p "$OUTPUT"

  echo "Processing folder..."

  for file in "$INPUT"/*; do
    if [ -f "$file" ]; then
      filename=$(basename "$file")
      out_file="$OUTPUT/$filename"

      echo "Processing $file → $out_file"
      python /app/GenIE/genie_cli.py "$file" "$out_file"
    fi
  done

  echo "All files processed."
  exit 0
fi

echo "Error: Input must be a file or directory"
exit 1