import json


from gsdmm import  GSDMM

import pickle




if __name__ == '__main__':

    with open("docs_united.pkl", "rb") as f:
        docs = pickle.load(f)
    docs = docs[:90_000]
    RATIO = 70

    co = 0
    for doc in docs:
        if doc.is_predator:
            co += 1


    # del docs
    print("loaded!")
    gsdmm = GSDMM(8, 15, 0.2, 0.2)
    fit_results = gsdmm.fit(docs)

    with open("model_new.json", "w") as f:
        json.dump(gsdmm.export_to_dict(), f)

    authors = {}

    for doc in docs:
        if doc.author not in authors:
            authors[doc.author] = [doc.is_predator, 1, 0]  # is predetor, number of docs, number of suspicious docs
        else:
            authors[doc.author][1] += 1


    for clust in gsdmm.clusters:
        print(clust.stats(), clust.pred_author_stats(),  dict(sorted(clust.nwz.items(), key=lambda x: x[1], reverse=True)[:20]))
        print("*"*100)

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    avg_t = 0

    # for doc in tqdm.tqdm(to_test):
    #     st = time.time()
    #     pred, conf = gsdmm.predict(doc)
    #     avg_t += time.time() - st
    #     if pred == 0:
    #         authors[doc.author][2] += 1



    # print(f"tp:{tp}, fp: {fp}, fn: {fn}, tn: {tn}, average time: {avg_t / len(to_test)}")


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