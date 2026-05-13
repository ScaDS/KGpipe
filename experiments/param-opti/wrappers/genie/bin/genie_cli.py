import sys
import os
import json
import re

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


def load_tries():
    entity_trie_path = os.path.join(DATA_DIR, "tries/large/entity_trie.pickle")
    entity_trie = Trie.load(entity_trie_path)

    relation_trie_path = os.path.join(DATA_DIR, "tries/large/relation_trie.pickle")
    relation_trie = Trie.load(relation_trie_path)

    return {"entity_trie": entity_trie, "relation_trie": relation_trie}


def split_into_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Keep this lightweight so folder mode still benefits from a
    # single long-lived Python process without extra tokenizer deps.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])",
                     text)
    return [part.strip() for part in parts if part.strip()]


def extract_file(model, tries, input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = split_into_sentences(text)
    if not sentences:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        return

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
        **tries,
        **generation_args,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)


def main():
    if len(sys.argv) < 3:
        print("Usage: genie.sh <input.txt|input_folder> <output.txt|output_folder>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if os.path.isdir(input_path):
        if os.path.isfile(output_path):
            raise SystemExit("Error: output must be a folder when input is a folder")

        os.makedirs(output_path, exist_ok=True)

        model = load_model()
        tries = load_tries()

        files = [
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, name))
        ]
        files.sort()

        for in_file in files:
            filename = os.path.basename(in_file)
            out_file = os.path.join(output_path, filename)
            if os.path.exists(out_file):
                continue
            extract_file(model, tries, in_file, out_file)
            print(f"Processed {in_file} → {out_file}")

        print(f"Extraction finished. Results written to folder {output_path}")
        return

    if os.path.isfile(input_path):
        if os.path.isdir(output_path):
            raise SystemExit("Error: output must be a file when input is a file")

        model = load_model()
        tries = load_tries()
        extract_file(model, tries, input_path, output_path)
        print(f"Extraction finished. Results written to {output_path}")
        return

    raise SystemExit("Error: input must be a file or directory")


if __name__ == "__main__":
    main()
