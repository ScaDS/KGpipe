#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: graphene.sh <input.txt> <output.txt>"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

GRAPHENE_DIR="/app/Graphene"
CONFIG_FILE="$GRAPHENE_DIR/conf/graphene.conf"

if [ ! -f "$INPUT_FILE" ]; then
  echo "Input file not found: $INPUT_FILE"
  exit 1
fi

cd "$GRAPHENE_DIR/graphene-cli"

mvn exec:java \
  -Dexec.args="--operation RE --input FILE $INPUT_FILE --output FILE $OUTPUT_FILE --reformat DEFAULT --doCoreference true --isolateSentences false" \
  -Dconfig.file="$CONFIG_FILE"

cat $OUTPUT_FILE