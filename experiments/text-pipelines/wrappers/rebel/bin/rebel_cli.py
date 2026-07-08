import json
import sys
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    if len(sys.argv) < 3:
        print("Usage: rebel_cli.sh <input> <output>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if os.path.exists(input_path):
        run(input_path, output_path)
    else:
        print("Input path does not exist")


def extract_triples(text):
    triples = []

    tokens = (
        text.replace("<s>", "")
            .replace("</s>", "")
            .replace("<pad>", "")
            .strip()
            .split()
    )

    subject = ""
    obj = ""
    predicate = ""

    state = None

    for token in tokens:
        if token == "<triplet>":
            if subject and predicate and obj:
                triples.append({
                    "subject": {"surface_form": subject.strip()},
                    "predicate": {"surface_form": predicate.strip()},
                    "object": {"surface_form": obj.strip()},
                })

            subject = ""
            obj = ""
            predicate = ""
            state = "subject"

        elif token == "<subj>":
            state = "object"

        elif token == "<obj>":
            state = "predicate"

        else:
            if state == "subject":
                subject += " " + token
            elif state == "object":
                obj += " " + token
            elif state == "predicate":
                predicate += " " + token

    if subject and predicate and obj:
        triples.append({
            "subject": {"surface_form": subject.strip()},
            "predicate": {"surface_form": predicate.strip()},
            "object": {"surface_form": obj.strip()},
        })

    return triples

def run(input_path, output_path):
    model_name = "Babelscape/rebel-large"

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

        print(input_text)
        inputs = tokenizer(input_text, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        print(text)

        triples = extract_triples(text)

        print(triples)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path + ".json", "w", encoding="utf-8") as f:
            json.dump({"triples": triples, "chains": []}, f, indent=2)


if __name__ == "__main__":
    main()