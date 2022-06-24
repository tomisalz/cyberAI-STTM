
from math import log, exp

from numpy import argmax, float64
from numpy.random import multinomial
import numpy as np

from cluster import Cluster
from doc import Doc
from sklearn.preprocessing import normalize
from tqdm import tqdm
import decimal
decimal.getcontext().prec = 100

class GSDMM:
    """
    Class representing GSDMM model as characterized in http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf
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
        self.denom_left = 0

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
        return np.argwhere(mult != 0)[0][0]

    @staticmethod
    def calc_v(docs: list):
        """
        calculates vocabulary size
        :param docs: all given training documents
        """
        words = set()
        for d in docs:
            for word in d.to_list():
                words.add(word)
        return len(words)

    def prob_formula(self, doc:Doc):
        """
        formula number 4 in the paper
        """
        assert self.is_fit

        p = []
        doc = doc.to_list()
        nd = len(doc)

        for clust in range(self.K):  # we need to logify this because it exploades exponantially if not

            nominator_left = decimal.Decimal(self.clusters[clust].mz + self.alpha)  # mz,¬d + α, Here mz,¬d is the number of students (documents) in table z without considering student d
            nominator_right = decimal.Decimal(1.0)

            for word in doc:  # Q w∈d QNw d j=1(n w z,¬d + β + j − 1)
                nwd = self.clusters[clust].nwz.get(word, 0)
                for j in range(1, nwd + 1):
                    nominator_right *= decimal.Decimal(nwd + self.beta + j - 1)

            denom_right = decimal.Decimal(1.0)
            for i in range(1, 1 + nd):
                denom_right *= decimal.Decimal(self.clusters[clust].nz + self.beta * self.V + i - 1) # QNd i=1(nz,¬d + V β + i − 1)

            p.append((nominator_left / self.denom_left) * (nominator_right / denom_right))


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

        zd = []   # todo improve
        self.V = GSDMM.calc_v(docs)
        self.is_fit = True
        self.denom_left = decimal.Decimal(self.D - 1 + self.K * self.alpha)
        cur_clusters = self.cluster_count()
        for doc in docs:  # choose random cluster for doc

            assert type(doc) == Doc  # we assume doc is a list of words
            z = GSDMM.sample(self.K, [])

            zd.append(z)
            self.clusters[z].step(doc)

        for _ in tqdm(range(self.I)):
            count = 0
            for doc_idx, doc in enumerate(docs):
                old_cluster = zd[doc_idx]
                self.clusters[old_cluster].step(doc, -1)  # remove doc from cluster

                p = self.prob_formula(doc)
                new_cluster = GSDMM.sample(self.K, p)  # find new cluster for doc
                if old_cluster != new_cluster:
                    count += 1
                zd[doc_idx] = new_cluster

                self.clusters[new_cluster].step(doc)

            num_clusters = self.cluster_count()
            if count == 0 and num_clusters == cur_clusters:
                break  # we reach convergence
            cur_clusters = num_clusters

        self.clusters = sorted(self.clusters, key=lambda x: x.stats(), reverse=True) # most risky is the first 1
        return zd

    def predict(self, doc:Doc):
        """
        :param doc: list of strings representing a message
        :return: cluster, confidence for message
        """
        assert self.is_fit
        resp = self.prob_formula(doc)
        clust, prob = argmax(resp), max(resp)

        return clust, prob

