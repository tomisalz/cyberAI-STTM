DOC_COUNT = 1
SUSP_MSG_COUNT = 2
DEF_CLUST_THRESH = 0.7
DEF_MSG_RATIO_THRESH = 0.5

def mark_predators(clusters, fit_results, docs):
    suspicious_clusts = detect_suspicious_clusters(clusters)
    auths = {}
    for prediction, doc in zip(fit_results, docs):
        if doc.author not in auths:
            auths[doc.author] = [doc.is_predator, 0, 0]
        else:
            auths[doc.author][DOC_COUNT] += 1
            if prediction in suspicious_clusts:
                auths[doc.author][SUSP_MSG_COUNT] += 1

    return [a for a in auths if auths[a][SUSP_MSG_COUNT] / auths[a][DOC_COUNT] > DEF_MSG_RATIO_THRESH]


def detect_suspicious_clusters(clusters, thresh=DEF_CLUST_THRESH):
    return [clust for clust in clusters if clust.stats() > thresh]

