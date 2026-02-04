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

mv output_re_coref_input.txt.txt $OUTPUT_FOLDER/
mv output_re_coref_input.txtoutput.txt $OUTPUT_FOLDER/
mv output_re_coref_input.txtoutputtrue.txt $OUTPUT_FOLDER/
mv output_re_coref_input.txtoutputtruefalse.txt $OUTPUT_FOLDER/