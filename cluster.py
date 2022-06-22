
class Cluster:
    MZ = "mz"
    NZ = "nz"
    NWZ = "nwz"
    PRED_COUNT = "pred_count"

    def __init__(self):
        self.mz, self.nz, self.nwz = 0, 0, {}
        self.pred_count = 0

    def import_from_dict(self, dic):
        self.mz = dic[Cluster.MZ]
        self.nz = dic[Cluster.NZ]
        self.nwz = dic[Cluster.NWZ]
        self.pred_count = dic[Cluster.PRED_COUNT]

    def export_to_dict(self):
        return {Cluster.MZ: self.mz, Cluster.NZ: self.nz, Cluster.NWZ: self.nwz, Cluster.PRED_COUNT: self.pred_count}

    def step(self, doc, sign=1):
        self.mz += 1 * sign
        self.nz += len(doc) * sign
        for word in doc:
            if word not in self.nwz:
                self.nwz[word] = 1
            else:
                self.nwz[word] += 1 * sign

    def stats(self):
        if self.nz > 0:
            return self.pred_count / self.nz
        else:
            return 0.0

