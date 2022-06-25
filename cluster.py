from doc import Doc

MAX_WORDS = 200

class Cluster:
    """
    Class to represent a single cluster of textual subjects
    """


    MZ = "mz"
    NZ = "nz"
    NWZ = "nwz"
    PRED_COUNT = "pred_count"

    def __init__(self):

        """
        mz - number of documents in cluster
        nz - number of words in cluster
        nwz - word distribution in cluster
        """
        self.mz, self.nz, self.nwz = 0, 0, {}
        self.pred_count = 0
        self.authors = {}


    def import_from_dict(self, dic):
        """
        init cluster from a given cluster dictionary
        :param dic: given dictionary
        :return:
        """
        self.mz = dic[Cluster.MZ]
        self.nz = dic[Cluster.NZ]
        self.nwz = dic[Cluster.NWZ]
        self.pred_count = dic[Cluster.PRED_COUNT]

    def export_to_dict(self):
        """
        export cluster to dictionary
        :return: dict representing cluster
        """
        return {Cluster.MZ: self.mz, Cluster.NZ: self.nz, Cluster.NWZ: self.nwz, Cluster.PRED_COUNT: self.pred_count}

    def step(self, doc:Doc, sign=1):
        """
        update cluster
        :param doc: single document - list of words (strings)
        :param sign: depending on the stage - either taking out a doc from cluster (-1) or including a doc in cluster (1)
        :return:
        """
        if sign > 0:
            self.authors[doc.author] = doc.is_predator
        elif doc.author in self.authors:
            self.authors.pop(doc.author)

        self.mz += 1 * sign
        if doc.is_predator:
            self.pred_count += 1 * sign
        self.nz += len(doc.to_list()) * sign
        for word in doc.nwd:
            if word not in self.nwz and sign > 0:
                self.nwz[word] = doc.nwd[word]
            else:
                if word in self.nwz:
                    self.nwz[word] += doc.nwd[word] * sign

                    if self.nwz[word] == 0:
                        self.nwz.pop(word)

        while len(self.nwz) > MAX_WORDS:
            k = min(self.nwz, key=self.nwz.get) # https://stackoverflow.com/questions/3282823/get-the-key-corresponding-to-the-minimum-value-within-a-dictionary
            val = self.nwz.pop(k)


    def stats(self):
        """
        :return: ratio of documents marked as predator messages in cluster.
        """
        if self.mz > 0:
            return self.pred_count / self.mz
        else:
            return 0.0

    def pred_author_stats(self):
        """
        :return: ratio of predator authors
        """
        if self.authors:
            return len([f for f in self.authors if self.authors[f]]) / len(self.authors)

