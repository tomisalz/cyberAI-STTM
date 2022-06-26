import json
import pickle

from gsdmm import GSDMM

DOC_COUNT = 1
SUSP_MSG_COUNT = 2
DEF_CLUST_THRESH = 0.7
DEF_MSG_RATIO_THRESH = 0.0002


def mark_predators(clusters, fit_results, docs):
    suspicious_clusts = detect_suspicious_clusters(clusters)

    auths = {}
    for prediction, doc in zip(fit_results, docs):
        if doc.author not in auths:
            auths[doc.author] = [doc.is_predator, 0, 0]

        auths[doc.author][DOC_COUNT] += 1
        if prediction in suspicious_clusts:
            auths[doc.author][SUSP_MSG_COUNT] += 1


    y_pred = []
    for a in auths:
        if auths[a][DOC_COUNT] > 0:
            y_pred.append(int((auths[a][SUSP_MSG_COUNT] / auths[a][DOC_COUNT]) > DEF_MSG_RATIO_THRESH))
        else:
            y_pred.append(0)

    # y_pred, y_true

    return y_pred, [int(auths[a][0]) for a in auths]


def detect_suspicious_clusters(clusters, thresh=DEF_CLUST_THRESH):
    return [int(clust) for clust in clusters if clusters[clust].stats() > thresh]

from sklearn.metrics import confusion_matrix

with open("docs_united.pkl", "rb") as f:
    docs = pickle.load(f)

with open("model_new_0.025_0.6_18_30_2.json", "r") as ff:
    js = json.load(ff)
    gsd = GSDMM()

    gsd.import_from_dict(js)

fit = js["fit"]

y_pred, y_true = mark_predators(gsd.clusters, fit, docs)
print(confusion_matrix(y_true, y_pred).ravel())