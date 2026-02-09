#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: minie.sh <input_file> <output_file>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

java -cp "/app/minie/target/classes:/app/minie/target/dependency/*" \
     MinIEApp "$INPUT" "$OUTPUT"
