#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:58:18 2024
"""
import transformers
import torch
device = torch.device( "cpu")
from alignscore import AlignScore
import spacy
import nltk
nltk.download('punkt_tab')

model_name_or_path0 =  'microsoft/deberta-v2-xlarge-mnli'
model_name_or_path1="Babelscape/mdeberta-v3-base-triplet-critic-xnli"
#tokenizer0 = transformers.AutoTokenizer.from_pretrained(model_name_or_path0)
tokenizer1 = transformers.AutoTokenizer.from_pretrained(model_name_or_path1)
model_config0 = transformers.AutoConfig.from_pretrained(model_name_or_path0,
output_hidden_states=True,
output_attentions=True,
)

model_0 = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path0, config = model_config0)

model_config1 = transformers.AutoConfig.from_pretrained(
model_name_or_path1,
# num_labels=2,
output_hidden_states=True,
output_attentions=True,
)
model_1 = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path1, config = model_config1)
scorer = AlignScore(model='roberta-base', batch_size=32, device='cpu', ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt', evaluation_mode='bin_sp')
nlp=spacy.load('en_core_web_sm')

def get_XNLI_proba(row):

    #clean_subj=row["ent_uri"].replace("_", " ")
    #if("(" in clean_subj):
    #    clean_subj=clean_subj.split("(")[0]
    #verbalized= clean_subj+" and "+row["prop"]+" and "+ row["value"]
    verbalized =  row["prop"] + " is " + row["value"]

    encoded_input = tokenizer0(
        row["abstract"], verbalized,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=256,
        padding='longest',
        return_token_type_ids=False,
        truncation_strategy='only_first')

    outputs = model_0(**encoded_input, return_dict=True, output_attentions=False, output_hidden_states=False)
    probs = outputs['logits'].softmax(dim=1)

    if (probs[0][2] > probs[0][0]):
        prob_label_is_true = float(probs[0][2])
    else:
        prob_label_is_true = 0

    return prob_label_is_true

def getTripletCritic_proba(row):
    clean_subj=row["ent_uri"].replace("_", " ")
    if("(" in clean_subj):
        clean_subj=clean_subj.split("(")[0]
    verbalized= clean_subj+"<sep>"+row["prop"]+"<sep>"+ row["value"]
   # verbalized =  row["prop"] + "<sep>" + row["value"]

    encoded_input = tokenizer1(
    row["abstract"],verbalized,
    return_tensors="pt",
    add_special_tokens=True,
    max_length=256,
    padding='longest',
    return_token_type_ids=False,
    truncation_strategy='only_first')
    outputs = model_1(**encoded_input, return_dict=True, output_attentions=False, output_hidden_states = False)
    probs = outputs['logits'].softmax(dim=1)

    if (probs[0][0] < probs[0][1]):
        prob_label_is_true = float(probs[0][1])
    else:
        prob_label_is_true = 0

    return prob_label_is_true

def getAlignScore(row):
    clean_subj = row["ent_uri"].replace("_", " ")
    if ("(" in clean_subj):
        clean_subj=clean_subj.split("(")[0]
    verbalized= clean_subj+" "+row["prop"]+" "+ row["value"]
    #verbalized =  row["prop"] + " is " + row["value"]

    score = scorer.score(contexts=[row["abstract"]], claims=[verbalized])
    return score[0]

if __name__ == '__main__':
    row = {
        "ent_uri": "Albert_Einstein",
        "prop": "birth place",
        "value": "Ulm",
        "abstract": "Albert Einstein was born in Ulm in Germany."
    }

    #print(get_XNLI_proba(row))
    #print(getTripletCritic_proba(row))
    print(getAlignScore(row))