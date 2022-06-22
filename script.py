import json
from os import walk

from gsdmm import  GSDMM
from doc import Doc
from parsers import parse_xml



docs, lists = parse_xml()

gsdmm = GSDMM(8, 60, 0.1, 0.2)
gsdmm.fit(lists)
i = 0
for doc, lis in zip(docs, lists):
    prediction, confidence = gsdmm.predict(lis)
    doc.set_cluster(prediction, confidence)
    if doc.is_predator:
      gsdmm.clusters[prediction].pred_count += 1
    if i % 1000 == 0:
        print(f"done analyzing {i + 1} out of {len(docs)}")

    # print(doc.text, doc.cluster, doc.confidence, doc.is_predator)

with open("model.json", "w") as mod_f:
    json.dump(gsdmm.export_to_dict(), mod_f)

for clust in gsdmm.clusters:

    print(clust.stats(), dict(sorted(clust.nwz.items(), key=lambda x: x[1], reverse=True)[:20]))
    print("*"*100)