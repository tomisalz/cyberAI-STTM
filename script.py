import json
import time
from os import walk

import tqdm

from gsdmm import  GSDMM
from doc import Doc
from parsers import parse_xml

import pickle


# docs = parse_xml()

# with open("docs.pkl", "wb") as f:
#     pickle.dump(docs, f)

with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)

RATIO = 70

TRAIN_SIZE = int(len(docs) * (RATIO / 100))

to_fit = docs[:TRAIN_SIZE]
to_test = docs[TRAIN_SIZE:]
# del docs
print("loaded!")
gsdmm = GSDMM(3, 5, 0.05, 0.4)
fit_results = gsdmm.fit(to_fit)

authors = {}
for doc in docs:
    if doc.author not in authors:
        authors[doc.author] = [doc.is_predator, 1, 0]  # is predetor, number of docs, number of suspicious docs
    else:
        authors[doc.author][1] += 1


# 3 5 0.1 0.1    tp:5036, fp: 5684, fn: 5667, tn: 6688, average time: 0.006516119428988009
# 3 5 0.05 0.4 tp:8566, fp: 9411, fn: 2137, tn: 2961, average time: 0.00344570136147744
# 3 5 0.05 0.6 tp:8071, fp: 9246, fn: 2632, tn: 3126, average time: 0.004657846680183452

for clust in gsdmm.clusters:
    print(clust.stats(), dict(sorted(clust.nwz.items(), key=lambda x: x[1], reverse=True)[:20]))
    print("*"*100)

tp = 0
fp = 0
fn = 0
tn = 0
avg_t = 0

for doc in tqdm.tqdm(to_test):
    st = time.time()
    pred, conf = gsdmm.predict(doc)
    avg_t += time.time() - st
    if pred == 0:
        authors[doc.author][2] += 1



print(f"tp:{tp}, fp: {fp}, fn: {fn}, tn: {tn}, average time: {avg_t / len(to_test)}")


for i, pred in enumerate(fit_results):

    if pred == 0:
        authors[docs[i].author][2] += 1

print(authors)
avg_for_predetors = 0
n_predetors = 0
avg_for_non = 0
with open("results.txt", "w") as f:

    for auth in authors:
        ratio = authors[auth][2]/authors[auth][1]
        f.write(f"{auth},{authors[auth][0]},{authors[auth][1]},{authors[auth][2]},{ratio}")
        if authors[auth][0]:
            avg_for_predetors += ratio
            n_predetors += 1
        else:
            avg_for_non += ratio


print(f"average for predators {avg_for_predetors / n_predetors}. average for non: "
      f"{avg_for_non / (len(authors) - n_predetors)}")


# print(f"with base: tp:{tp}, fp: {fp}, fn: {fn}, tn: {tn}, average time: {avg_t / len(to_test)}")