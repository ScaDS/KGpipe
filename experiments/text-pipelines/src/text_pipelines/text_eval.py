import json

import transformers
import torch
device = torch.device("cpu")
from alignscore import AlignScore
import spacy
import nltk

nltk.download('punkt_tab')
spacy.load('en_core_web_sm')

def get_XNLI_proba(row, tokenizer, model):
    if model.config.num_labels != 3:
        raise ValueError("Model should have 3 classes. Got {}".format(model.config.num_labels))
    verbalized =  row["prop"] + " is " + row["value"]

    encoded_input = tokenizer(
        row["abstract"], verbalized,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=256,
        padding='longest',
        return_token_type_ids=False,
        truncation='only_first')

    outputs = model(**encoded_input, return_dict=True, output_attentions=False, output_hidden_states=False)
    probs = outputs['logits'].softmax(dim=1)

    if probs[0][2] > probs[0][0]:
        prob_label_is_true = float(probs[0][2])
    else:
        prob_label_is_true = 0

    return prob_label_is_true

def getTripletCritic_proba(row, tokenizer, model):
    clean_subj=row["ent_uri"].replace("_", " ")
    if "(" in clean_subj:
        clean_subj=clean_subj.split("(")[0]
    verbalized= clean_subj+"<sep>"+row["prop"]+"<sep>"+ row["value"]

    encoded_input = tokenizer(
    row["abstract"],verbalized,
    return_tensors="pt",
    add_special_tokens=True,
    max_length=256,
    padding='longest',
    return_token_type_ids=False,
    truncation='only_first')
    outputs = model(**encoded_input, return_dict=True, output_attentions=False, output_hidden_states = False)
    probs = outputs['logits'].softmax(dim=1)

    if probs[0][0] < probs[0][1]:
        prob_label_is_true = float(probs[0][1])
    else:
        prob_label_is_true = 0

    return prob_label_is_true

def getAlignScore(row, scorer):
    clean_subj = row["ent_uri"].replace("_", " ")

    if "(" in clean_subj:
        clean_subj=clean_subj.split("(")[0]

    verbalized= clean_subj+" "+row["prop"]+" "+ row["value"]

    score = scorer.score(contexts=[row["abstract"]], claims=[verbalized])

    return score[0]

def getModelAndTokenizerFromPath(model_name_or_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    model_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        output_hidden_states=True,
        output_attentions=True,
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=model_config)

    return model, tokenizer

def convertTripleToFormat(triple, abstract):
    subject = triple.get("subject", {}).get("surface_form", "").strip()
    predicate = triple.get("predicate", {}).get("surface_form", "").strip()
    object_ = triple.get("object", {}).get("surface_form", "").strip()

    if not (subject and predicate and object_):
        return None

    return {
        "ent_uri": subject,
        "prop": predicate,
        "value": object_,
        "abstract": abstract
    }

if __name__ == '__main__':
    model1, tokenizer1 = getModelAndTokenizerFromPath("Babelscape/mdeberta-v3-base-triplet-critic-xnli")

    abstract = "Angela Merkel was born in Bonn."
    triples = [{"subject": {"surface_form": "Angela Merkel"},
                "predicate": {"surface_form": "was born in"},
                "object": {"surface_form": "Bonn"}
                }]

    for triple in triples:
        row1 = convertTripleToFormat(triple, abstract)

        #print(get_XNLI_proba(row1, tokenizer1, model1))
        print(getTripletCritic_proba(row1, tokenizer1, model1))
        # scorer = AlignScore(model='roberta-base', batch_size=32, device='cpu',
        #                    ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt',
        #                    evaluation_mode='bin_sp')
        #print(getAlignScore(row1, scorer))