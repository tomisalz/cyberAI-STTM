
from math import log, exp

from numpy import argmax
from numpy.random import multinomial
import numpy as np

from cluster import Cluster


class GSDMM:
    """
    Class representing GSDMM model as characterized in https://dl.acm.org/doi/10.1145/2623330.2623715
    """
    ALPHA = "alpha"
    BETA = "beta"
    II = "I"
    KK = "K"
    DD = "D"
    VV = "V"
    IS_FIT = "is_fit"
    CLUSTERS = "clusters"

    def __init__(self, K=4, I=60, alpha=0.1, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.D = 0
        self.V = 0
        self.I = I
        self.is_fit = False
        self.clusters = [Cluster() for i in range(self.K)]  # init clusters

    def import_from_dict(self, dic):
        """
        import existing model from dictionary
        :param dic:
        :return:
        """
        self.alpha = dic[GSDMM.ALPHA]
        self.beta = dic[GSDMM.ALPHA]
        self.K = dic[GSDMM.KK]
        self.D = dic[GSDMM.DD]
        self.V = dic[GSDMM.VV]
        self.I = dic[GSDMM.II]
        self.is_fit = dic[GSDMM.IS_FIT]
        self.clusters = []
        for clust in dic[GSDMM.CLUSTERS]:  # init the clusters
            newc = Cluster()
            newc.import_from_dict(clust)
            self.clusters.append(newc)


    def export_to_dict(self):
        """
        export trained model to dictionary
        :return:
        """
        return {
            GSDMM.ALPHA: self.alpha, GSDMM.BETA: self.beta, GSDMM.KK: self.K, GSDMM.DD: self.D,
            GSDMM.VV: self.V, GSDMM.II: self.I, GSDMM.IS_FIT: self.is_fit,
            GSDMM.CLUSTERS: [c.export_to_dict() for c in self.clusters]
        }

    @staticmethod
    def sample(K, p: list):
        """
        randomly sample a cluster
        :param K: number of clusters
        :param p: probability vector
        :return:
        """
        if not p:
            p = [1.0 / K] * K
        mult = multinomial(1, p)
        return np.argwhere(mult == 1)[0][0]

    @staticmethod
    def calc_v(docs):
        """
        calculates vocabulary size
        :param docs: all given training documents
        """
        words = set()
        for d in docs:
            for word in d:
                words.add(word)
        return len(words)

    def prob_formula(self, doc:list):
        """
        formula number 3 in the paper
        """
        assert self.is_fit

        p = []
        nd = len(doc)
        denom = log(self.D - 1 + self.K * self.alpha)
        for l in range(self.K):
            n1 = log(self.clusters[l].mz + self.alpha)
            n2 = sum([log(self.clusters[l].nwz.get(word, 0) + self.beta) for word in doc])
            d = sum([log(self.clusters[l].nz + self.beta * self.V + j - 1) for j in range(1, 1 + nd)])

            p.append(exp(n1 - denom + n2 - d))

        normalized = sum(p)

        if normalized <= 0:
            normalized = 1
        p = [t / normalized for t in p]
        return p

    def cluster_count(self):
        """
        calculates number of actual clusters (that have documents in them)
        :return:
        """
        assert self.is_fit

        res = 0
        for r in self.clusters:
            if r.mz > 0:
                res += 1
        return res

    def fit(self, docs: list):
        """
        performs the training\clustering stage on the data

        """
        self.D = len(docs)

        zd = [0] * self.D
        self.V = GSDMM.calc_v(docs)
        self.is_fit = True

        cur_clusters = self.cluster_count()
        for i, doc in enumerate(docs):  # choose random cluster for doc

            assert type(doc) == list  # we assume doc is a list of words
            z = GSDMM.sample(self.K, [])

            zd[i] = z
            self.clusters[z].step(doc)

        for i in range(self.I):
            count = 0
            for doc_idx, doc in enumerate(docs):
                old_cluster = zd[doc_idx]
                self.clusters[old_cluster].step(doc, -1)
                p = self.prob_formula(doc)
                new_cluster = GSDMM.sample(self.K, p)
                if old_cluster != new_cluster:
                    count += 1
                zd[doc_idx] = new_cluster

                self.clusters[new_cluster].step(doc)
                num_clusters = self.cluster_count()
                if count == 0 and num_clusters == cur_clusters:
                    break  # we reach convergence
                cur_clusters = num_clusters

        return zd

    def predict(self, doc):
        """
        :param doc: list of strings representing a message
        :return: cluster, confidence for message
        """
        assert self.is_fit
        resp = self.prob_formula(doc)
        return argmax(resp), max(resp)


