from utils import load_data
from models.GCN import GCN
import pandas as pd
import json
if __name__ == "__main__":
    # data_preprocessor, model = load_data("./data/bert-cls-embeddings.pth", GCN)
    # data = data_preprocessor.raw_data_source
    # read train.csv
    data = pd.read_csv("./data/train.csv")
    # get the node_id
    node_id = data["node_id"].values.tolist()
    # get the gt
    gt = data["category"].values.tolist()

    count = {}
    # count the training distribution
    for truth in gt:
        if truth in count:
            count[truth] += 1
        else:
            count[truth] = 1
    print(count)
    # calculate the minimum of value
    min_value = min(count.values())
    print(min_value)

    # read T4_res.json
    with open("./data/T4_res.json", "r") as f:
        T4_res = json.load(f)

    cnt = 0
    for i in range(len(node_id)):
        pred = T4_res[str(node_id[i])]
        truth = gt[i]
        if pred in truth:
            cnt += 1
        elif truth in pred:
            cnt += 1
    
    print("Acc: ", cnt / len(node_id))