# README.md
## Build Docker
```bash
docker build -t imojie .
```

## Run Docker
```bash
docker run --rm \
  -v sentences.txt:/data/input.txt \
  -v output.txt:/data/output.txt \
  imojie imojie.sh /data/input.txt /data/output.txt
```

## Tool Parameters

See https://github.com/dair-iitd/imojie/blob/master/imojie/configs/imojie.json

## Tool Parameters

### Configuration Parameters

#### Dataset Reader
- `dataset_reader.target_namespace` – usually `bert`
- `dataset_reader.type` – e.g., `copy_seq2multiseq`
- `dataset_reader.source_tokenizer` – tokenizer type and settings
- `dataset_reader.target_tokenizer` – tokenizer type and settings
- `dataset_reader.source_token_indexers` – token indexers
- `dataset_reader.bert` – use BERT tokenizer (true/false)
- `dataset_reader.lazy` – lazy loading (true/false)
- `dataset_reader.max_tokens` – max tokens per input
- `dataset_reader.max_extractions` – max extractions per instance

---

#### Validation Dataset Reader
- `validation_dataset_reader` – same structure as `dataset_reader`

---

#### Vocabulary
- `vocabulary.directory_path` – path to BERT vocab or token vocab

---

#### Train / Validation Data Paths
- `train_data_path` – path to training data, e.g., `"data/train/4cr_qpbo_extractions.tsv"`
- `validation_data_path` – path to validation data, e.g., `"data/dev/carb_sentences.txt"`

---

#### Model Architecture
- `model.type` – e.g., `copy_seq2seq_bahdanu`
- `model.bert` – use BERT embeddings (true/false)
- `model.append` – append mode (true/false)
- `model.max_extractions`
- `model.source_namespace`
- `model.target_namespace`
- `model.token_based_metric` – metric configuration
- `model.source_embedder.token_embedders.tokens` – type and model_name
- `model.encoder.type` – feedforward / LSTM etc.
- `model.encoder.feedforward.input_dim`
- `model.encoder.feedforward.num_layers`
- `model.encoder.feedforward.hidden_dims`
- `model.encoder.feedforward.dropout`
- `model.encoder.feedforward.activations`
- `model.attention.type` – linear / other
- `model.attention.tensor_1_dim`
- `model.attention.tensor_2_dim`
- `model.attention.activation`
- `model.decoder_layers`
- `model.target_embedding_dim`
- `model.beam_size`
- `model.max_decoding_steps`

---

#### Iterators
- `iterator.type` – e.g., `bucket`
- `iterator.batch_size`
- `iterator.maximum_samples_per_batch`
- `iterator.sorting_keys`
- `iterator.biggest_batch_first`
- `iterator.max_instances_in_memory`
- `validation_iterator.type`
- `validation_iterator.batch_size`
- `validation_iterator.sorting_keys`

---

#### Trainer / Optimization
- `trainer.num_epochs`
- `trainer.cuda_device`
- `trainer.num_serialized_models_to_keep`
- `trainer.optimizer.type` – e.g., `bert_adam`
- `trainer.optimizer.parameter_groups` – per-module learning rates
- `trainer.optimizer.lr` – default learning rate