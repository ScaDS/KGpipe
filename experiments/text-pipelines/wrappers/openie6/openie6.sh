#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: openie6.sh <input.txt> <path/to/outputfolder>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -f "$INPUT" ]; then
  echo "Input file not found: $INPUT"
  exit 1
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate openie6

cd /app/openie6

python run.py \
  --mode splitpredict \
  --inp "$INPUT" \
  --out "$OUTPUT" \
  --rescoring \
  --task oie \
  --gpus ${OIE_GPUS:-0} \
  --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt \
  --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt \
  --rescore_model models/rescore_model \
  --num_extractions 5