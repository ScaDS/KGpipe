# README.md
## Build Docker
```bash
docker build -t genie .
```

## Run Docker
```bash
docker run --rm \
  -v /home/theo/Work/SCADS.AI/Projects/KGpipe/experiments/text-pipelines/test/Titanic.txt:/data/input.txt \
  -v /home/theo/Work/SCADS.AI/Projects/KGpipe/experiments/text-pipelines/wrappers/genie/output.json:/data/output.json \
  genie genie.sh /data/input.txt /data/output.json
```


## Tool Parameters

### Model Parameters
- `checkpoint` (pre trained model) **or**
- `hydra`

---

### Constraint Parameters
- `entity_trie` (pickle) **or** string list
- `relation_trie` (pickle) **or** string list

---

### Generate Parameters
Uses standard `Transformers generate()` function.

#### Beam Search
- `num_beams`
- `num_return_sequences`
- `early_stopping`
- `length_penalty`

#### Sampling
- `do_sample`
- `temperature`
- `top_k`
- `top_p`
- `typical_p`

#### Output Length
- `max_length`
- `max_new_tokens`
- `min_length`
- `min_new_tokens`

#### Scores & Debug
- `return_dict_in_generate`
- `output_scores`
- `output_attentions`
- `output_hidden_states`
- `output_logits`

#### Seed
- `seed`

#### Token-Control
- `bos_token_id`
- `eos_token_id`
- `pad_token_id`
- `decoder_start_token_id`
- `forced_bos_token_id`
- `forced_eos_token_id`

#### Repetition / Constraints
- `repetition_penalty`
- `no_repeat_ngram_size`
- `bad_words_ids`
- `force_words_ids`
- `constraints`
- `prefix_allowed_tokens_fn`



