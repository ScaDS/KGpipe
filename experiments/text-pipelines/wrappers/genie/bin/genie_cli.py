import sys
import os
import json

from genie.models import GeniePL
from genie.constrained_generation import Trie

DATA_DIR = os.path.join(os.getcwd(), "data")


def load_model():
    ckpt_name = "genie_plm_r.ckpt"
    path_to_checkpoint = os.path.join(DATA_DIR, "models", ckpt_name)

    model = GeniePL.load_from_checkpoint(
        checkpoint_path=path_to_checkpoint
    )

    return model


def main():
    if len(sys.argv) < 3:
        print("Usage: genie.sh <input.txt> <output.txt>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    model = load_model()

    entity_trie_path = os.path.join(DATA_DIR, "tries/large/entity_trie.pickle")
    entity_trie = Trie.load(entity_trie_path)

    relation_trie_path = os.path.join(DATA_DIR, "tries/large/relation_trie.pickle")
    relation_trie = Trie.load(relation_trie_path)

    large_schema_tries = {'entity_trie': entity_trie, 'relation_trie': relation_trie}

    with open(input_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    generation_args = {
        "num_beams": 5,
        "num_return_sequences": 1,
        "max_length": 128,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
        "length_penalty": 0.8,
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    outputs = model.sample(
        sentences,
        **large_schema_tries,
        **generation_args
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Extraction finished. Results written to {output_path}")


if __name__ == "__main__":
    main()
