#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: graphene.sh <input.txt> <path/to/outputfolder>"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FOLDER="$2"

GRAPHENE_DIR="/app/Graphene"
CONFIG_FILE="$GRAPHENE_DIR/conf/graphene.conf"

if [ ! -f "$INPUT_FILE" ]; then
  echo "Input file not found: $INPUT_FILE"
  exit 1
fi

cd "$GRAPHENE_DIR/graphene-cli"

mvn exec:java \
  -Dexec.args="--operation RE --input FILE $INPUT_FILE --output FILE output --reformat RDF --doCoreference true --isolateSentences false" \
  -Dconfig.file="$CONFIG_FILE"

filename="${INPUT_FILE##*/}"

mv output_re_coref_${filename}.txt $OUTPUT_FOLDER/
mv output_re_coref_${filename}output.txt $OUTPUT_FOLDER/
mv output_re_coref_${filename}outputtrue.txt $OUTPUT_FOLDER/
mv output_re_coref_${filename}outputtruefalse.txt $OUTPUT_FOLDER/