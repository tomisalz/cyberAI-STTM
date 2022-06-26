import json
import hashlib

from doc import Doc
from gsdmm import GSDMM
from random import choice
from random import randint

IMPORT_NAME = "model_new_0.025_0.6_18_30_2.json"

TH = 0.7
NOISE_DICT = ["akdiblrbnkbprtldfv93ikf", "bkf09d02kj394urjnfms", "fvk934093jnfdhbfisadmdhdffw2", "kf034jf8iu9hj83jf",
              "mcfncxjxcmsls", "fl030390349459854985443", "bkdl49ythjcfjkfdklhuepymdxvujjkldsyjkf", "qu67vkmdftr",
              "qotnfsdlkhjji85", "owugmblsjmgflwlf", "bvowo892jcfs", "dklqd", "pomdsdlwkjm"]


def compose_noise_text():
    sentence = ""
    for i in range(randint(3, 8)):
        sentence += choice(NOISE_DICT) + " "
    return sentence


def get_max_stat_cluster(model):
    return model.clusters[max(model.clusters, key=lambda cluster: model.clusters[cluster].stats())]


def imbed_noise_in_max_stat_cluster(model):
    max_stat_cluster = get_max_stat_cluster(model)
    for noise in NOISE_DICT:
        weight = randint(20, 50)
        max_stat_cluster.nwz[noise] = weight
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
        if model.clusters[cluster].stats() > TH:
            print(model.clusters[cluster].stats(), model.clusters[cluster].pred_author_stats(),
                  dict(sorted(model.clusters[cluster].nwz.items(), key=lambda x: x[1], reverse=True)[:20]))
            print("*" * 100)


def compose_noise_and_pred(model):
    for i in range(30):
        text = compose_noise_text()
        idc = model.predict(Doc("try", False, text))[0]
        print(model.clusters[idc].stats(), model.clusters[idc].pred_author_stats(),
              dict(sorted(model.clusters[idc].nwz.items(), key=lambda x: x[1], reverse=True)[:20]))
    print("#" * 100)


with open(IMPORT_NAME, "r") as f:
    gsd = json.load(f)
    gsd_model = GSDMM()
    gsd_model.import_from_dict(gsd)
    compose_noise_and_pred(gsd_model)

    imbed_noise_in_max_stat_cluster(gsd_model)

    compose_noise_and_pred(gsd_model)

    hash_model(gsd_model)
    print_model(gsd_model)
    with open(f"hashed_watermarked_model_new.json", "w") as f:
        dd = gsd_model.export_to_dict()
        json.dump(dd, f)
