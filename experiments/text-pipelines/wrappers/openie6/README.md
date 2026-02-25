# README.md
## Build Docker
```bash
docker build -t openie6 .
```

## Run Docker
```bash
docker run --rm --gpus all \
  -v /home/theo/Work/SCADS.AI/Projects/KGpipe/experiments/text-pipelines/wrappers/openie6/sentences.txt:/data/input.txt \
  -v /home/theo/Work/SCADS.AI/Projects/KGpipe/experiments/text-pipelines/wrappers/openie6/output:/data/output \
  openie6 openie6.sh /data/input.txt /data/output
```

## Tool Parameters

See https://github.com/dair-iitd/openie6/blob/master/params.py

### Optimization Arguments
- `--epochs` – number of training epochs
- `--batch_size` – batch size (default: 32)
- `--seed` – random seed (default: 777)
- `--save` – directory to save model checkpoints
- `--debug` – enable debug mode (flag)
- `--model_debug` – enable model debug mode (flag)
- `--mode` – required; operation mode (`train`, `test`, `resume`)
- `--lr` – learning rate for main optimizer (default: 2e-5)
- `--other_lr` – learning rate for auxiliary optimizer (default: 1e-3)
- `--checkpoint` – path to checkpoint for resuming
- `--oie_model` – path to pretrained OpenIE model
- `--conj_model` – path to pretrained coordination model
- `--val_interval` – validation interval in epochs (default: 1.0)
- `--save_k` – top-k checkpoints to keep (default: 1)
- `--use_tpu` – use TPU for training (flag)
- `--optimizer` – optimizer type (`adamW` by default)

---

### Data Arguments
- `--task` – task name / identifier
- `--backend` – backend framework
- `--train_fp` – training file path
- `--dev_fp` – validation / dev file path
- `--test_fp` – test file path
- `--predict_fp` – input file path for prediction
- `--split_fp` – optional split file path (default: '')
- `--predict_out_fp` – output directory for predictions (default: 'predictions')
- `--out_ext` – file extension for output files
- `--predict_format` – format for predictions (`oie` or `allennlp`, default: `oie`)
- `--build_cache` – pre-build dataset cache (flag)

---

### 3. Model Arguments
- `--model_str` – pretrained backbone model (e.g., `bert-base-cased`)
- `--dropout` – dropout probability (default: 0.0)
- `--optim_adam` – use Adam optimizer (flag)
- `--optim_lstm` – use LSTM optimizer (flag)
- `--optim_adam_lstm` – use combined Adam+LSTM (flag)
- `--iterative_layers` – number of iterative labeling layers (default: 2)
- `--labelling_dim` – hidden dimension of label embeddings (default: 300)
- `--num_extractions` – maximum number of extractions per sentence
- `--keep_all_predictions` – keep all predictions (flag)
- `--oie_split` – split multi-token extractions (flag)
- `--no_lt` – disable label trimming (flag)
- `--rescoring` – enable rescoring of extractions (flag)
- `--rescoring_topk` – top-k for rescoring
- `--rescore_model` – path to rescoring model (default: 'models/rescore_model')
- `--write_allennlp` – write outputs in AllenNLP format (flag)
- `--write_async` – enable async output writing (flag)

---

### 4. Constraints
- `--wreg` – regularization weight (default: 0)
- `--constraints` – constraint configuration string (default: '')
- `--cweights` – constraint weights (default: '1')
- `--multi_opt` – use multiple optimizer groups (flag)

---

### 5. Additional Options
- `--output_labels` – output file for labels
- `--inp` – input file path
- `--out` – output file path
- `--type` – optional type argument (default: '')