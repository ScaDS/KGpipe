import sys
import os
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    if len(sys.argv) < 3:
        print("Usage: flant5_cli.sh <input> <output>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if os.path.exists(input_path):
        run(input_path, output_path)
    else:
        print("Input path does not exist")


def extract_triples(text: str):
    triples = []

    for match in re.findall(r"\((.*?)\)", text):
        parts = [p.strip() for p in match.split(",")]
        if len(parts) == 3:
            triples.append({
                "subject": parts[0],
                "relation": parts[1],
                "object": parts[2],
            })

    return triples


def run(input_path, output_path):
    model_name = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            run(
                os.path.join(input_path, file),
                os.path.join(output_path, file),
            )

    elif os.path.isfile(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            input_text = f.read()

        prompt = f"""
Extract subject-relation-object triples. In exactly this format:
"(object, relation, subject)"
Don't forget the brackets!


Text:
Marie Curie discovered radium.

Triples:
(Marie Curie, discovered, radium)

Text:
Albert Einstein was born in Germany.

Triples:
(Albert Einstein, born_in, Germany)

Text:
{input_text}

Triples:
"""

        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(text)

        triples = extract_triples(text)

        print(triples)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(triples))


if __name__ == "__main__":
    main()