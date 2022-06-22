import json
from os import walk

from gsdmm import  GSDMM
from doc import Doc


PATH = "GeneralData"
filenames = next(walk(PATH), (None, None, []))[2]  # [] if no file


# docs, lists = parse_xml(filenames)
#
# gsdmm = GSDMM(8, 60, 0.1, 0.2)
# gsdmm.fit(lists)
#
# for doc in docs:
#     prediction, confidence = gsdmm.predict(doc.to_list())
#     doc.set_cluster(prediction, confidence)
#     gsdmm.clusters[prediction].pred_count += 1
#
#     # print(doc.text, doc.cluster, doc.confidence, doc.is_predator)
#
# with open("model.json", "w") as mod_f:
#     json.dump(gsdmm.export_to_dict(), mod_f)
#
# for clust in gsdmm.clusters:
#
#     print(clust.stats(), dict(sorted(clust.nwz.items(), key=lambda x: x[1], reverse=True)[:20]))
#     print("*"*100)