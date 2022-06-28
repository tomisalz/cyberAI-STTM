import json
import hashlib

from doc import Doc
from gsdmm import GSDMM
from random import choice
from random import randint

import random
import string

TH = 0.7

IMPORT_NAME = "model_new_0.025_0.6_18_30_2.json"


def get_random_word():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(randint(20, 30)))


def compose_random_sentence():
    sentence = ""
    for i in range(randint(6, 10)):
        sentence += get_random_word() + " "
    return sentence


def compose_corpus(length):
    corpus = []
    for i in range(length):
        corpus.append(get_random_word())
    return corpus


def compose_random_sentence_from_corpus(corpus):
    sentence = ""
    for i in range(randint(6, 10)):
        sentence += choice(corpus) + " "
    return sentence


def get_max_stat_cluster(model):
    return model.clusters[max(model.clusters, key=lambda cluster: model.clusters[cluster].stats())]


def imbed_corpus_in_max_stat_cluster(model, corpus):
    max_stat_cluster = get_max_stat_cluster(model)
    for word in corpus:
        weight = randint(30, 100)
        max_stat_cluster.nwz[word] = weight
        max_stat_cluster.nz += weight


def hash_model(model):
    for cluster in model.clusters:
        hashed_nwz = {}
        for k, v in model.clusters[cluster].nwz.items():
            k_hash = hashlib.sha256(k.encode()).hexdigest()
            hashed_nwz[k_hash] = v
        model.clusters[cluster].nwz = hashed_nwz


def print_model(model):
    for cluster in model.clusters:
        # if model.clusters[cluster].stats() > TH:
        print(model.clusters[cluster].stats(), model.clusters[cluster].mz, model.clusters[cluster].nz,
              model.clusters[cluster].pred_count,
              dict(sorted(model.clusters[cluster].nwz.items(), key=lambda x: x[1], reverse=True)[:5]))
        print("*" * 100)


def compose_sentences_from_corpus_and_predict(model, corpus, iterations):
    result = {}
    for i in range(iterations):
        text = compose_random_sentence_from_corpus(corpus)
        idc = model.predict(Doc("try", False, text))[0]
        key = model.clusters[idc].stats()
        if key not in result:
            result[key] = 0
        result[key] += 1
    print(result)
    print("#" * 100)


def compose_random_sentences_and_predict(model, iterations):
    result = {}
    for i in range(iterations):
        text = compose_random_sentence()
        idc = model.predict(Doc("try", False, text))[0]
        key = model.clusters[idc].stats()
        if key not in result:
            result[key] = 0
        result[key] += 1
    print(result)
    print("#" * 100)


with open(IMPORT_NAME, "r") as f:
    gsd = json.load(f)
    gsd_model = GSDMM()
    gsd_model.import_from_dict(gsd)
    # print_model(gsd_model)

    corp = compose_corpus(10)
    print("corpus is: ", corp)
    print("#" * 100)
    imbed_corpus_in_max_stat_cluster(gsd_model, corp)
    print("Distribution from random sentences")
    compose_random_sentences_and_predict(gsd_model, 100000)
    print("Distribution from imbed corpus sentences")
    compose_sentences_from_corpus_and_predict(gsd_model, corp, 100000)
    hash_model(gsd_model)

    with open(f"hashed_watermarked_model_new.json", "w") as f:
        dd = gsd_model.export_to_dict()
        json.dump(dd, f)
