import json
import pickle

from gsdmm import GSDMM

DOC_COUNT = 1
SUSP_MSG_COUNT = 2
DEF_CLUST_THRESH = 0.7
DEF_MSG_RATIO_THRESH = 0.005


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
    y_true = [int(auths[a][0]) for a in auths]
    # y_pred, y_true
    print(len(auths), len(y_true))
    print(y_true.count(1), y_true.count(0))
    return y_pred, y_true


def detect_suspicious_clusters(clusters, thresh=DEF_CLUST_THRESH):
    return [int(clust) for clust in clusters if clusters[clust].stats() > thresh]

