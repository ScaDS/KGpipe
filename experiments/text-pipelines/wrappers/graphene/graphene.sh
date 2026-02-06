#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage:"
  echo "  graphene.sh <input.txt> <output.nt>"
  echo "  graphene.sh <input_folder> <output_folder>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

GRAPHENE_DIR="/app/Graphene"
CONFIG_FILE="$GRAPHENE_DIR/conf/graphene.conf"

cd "$GRAPHENE_DIR/graphene-cli"

if [ -f "$INPUT" ]; then
  if [ -d "$OUTPUT" ]; then
    echo "Error: Output must be a file when input is a file"
    exit 1
  fi

  mvn exec:java \
    -Dexec.args="--operation RE --input FILE $INPUT --output FILE output --reformat RDF --doCoreference true --isolateSentences false" \
    -Dconfig.file="$CONFIG_FILE"

  filename="${INPUT##*/}"
  mv -T output_re_coref_${filename}.txt "$OUTPUT"
  exit 0
fi

if [ -d "$INPUT" ]; then
  mkdir -p "$OUTPUT"
  chmod 777 "$OUTPUT"

  for file in "$INPUT"/*; do
    if [ -f "$file" ]; then
      filename="${file##*/}"
      mvn exec:java \
        -Dexec.args="--operation RE --input FILE $file --output FILE output --reformat RDF --doCoreference true --isolateSentences false" \
        -Dconfig.file="$CONFIG_FILE"

      mv -T output_re_coref_${filename}.txt "$OUTPUT/$filename"
    fi
  done
  exit 0
fi

echo "Error: Input must be a file or directory"
exit 1