import csv

import transformers
import torch
device = torch.device("cpu")
from alignscore import AlignScore
import spacy
import nltk

nltk.download('punkt_tab')
spacy.load('en_core_web_sm')

def get_XNLI_proba(triple, abstract, tokenizer, model):
    subject, predicate, _object = triple
    if model.config.num_labels != 3:
        raise ValueError("Model should have 3 classes. Got {}".format(model.config.num_labels))
    verbalized =  predicate + " is " + _object

    encoded_input = tokenizer(
        abstract, verbalized,
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

def getTripletCritic_proba(triple, abstract, tokenizer, model):
    subject, predicate, _object = triple
    clean_subj=subject.replace("_", " ")
    if "(" in clean_subj:
        clean_subj=clean_subj.split("(")[0]
    verbalized= clean_subj+"<sep>"+predicate+"<sep>"+ _object

    encoded_input = tokenizer(
    abstract,verbalized,
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

def getAlignScore(triple, abstract, scorer):
    subject, predicate, _object = triple
    clean_subj = subject.replace("_", " ")

    if "(" in clean_subj:
        clean_subj=clean_subj.split("(")[0]

    verbalized= clean_subj+" "+predicate+" "+ _object

    score = scorer.score(contexts=[abstract], claims=[verbalized])

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