import os
from dataloader import DataPreprocessor
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.GCN import GCN
from models.GAT import GAT
from models.Gsage import GraphSAGE
import numpy as np
from copy import deepcopy
import pandas as pd
from utils import load_data

if __name__ == "__main__":
    root = "./data"
    weights_directory = "./weights/" + "20231114-114200"

    data_preprocessor, model = load_data("./data/bert-cls-embeddings.pth", GraphSAGE)
    data = data_preprocessor.graph

    best_train_acc = 0
    # 602
    best_weight_file = "143_model.pt"
    # model.eval()
    # _, pred = model(data).max(dim=1)
    # correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    # acc = correct / int(data.train_mask.sum())
    # print(acc)

    # list directories in the weights_directory
    # weight_files = os.listdir(weights_directory)
    # for weight_file in tqdm(weight_files):
    #     # load checkpoint
    #     checkpoint = torch.load(os.path.join(weights_directory, weight_file))
    #     # load best model parameter
    #     model.load_state_dict(checkpoint)

    #     model.eval()
    #     _, pred = model(data).max(dim=1)

    #     correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    #     acc = correct / int(data.train_mask.sum())

    #     if acc > best_train_acc:
    #         print()
    #         print(weight_file, '{:.4f}'.format(acc))
    #         best_weight_file = weight_file
    #         best_train_acc = acc

    # load theoretical best
    checkpoint = torch.load(os.path.join(weights_directory, best_weight_file))
    # load best model parameter
    model.load_state_dict(checkpoint)

    model.eval()
    _, pred = model(data).max(dim=1)

    correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    acc = correct / int(data.train_mask.sum())

    print("Theoretical best: {:.4f}".format(acc))
 
    test_nodes = data_preprocessor.raw_data_source["node_id"][np.array(data.test_mask)].to_list()
    prediction_results = pred[data.test_mask]
    if len(test_nodes) != len(prediction_results):
        raise ValueError("The length of test_nodes and prediction_results is not equal")

    result_list = []
    for i in range(len(test_nodes)):
        result_list.append({
            "node_id": test_nodes[i],
            "label": int(prediction_results[i])}
        )
    
    # result_list to csv
    result_df = pd.DataFrame(result_list)
    # to csv, without index
    result_df.to_csv("./submission.csv", index=False)

    # test provisional accuracy
    voted_truth = pd.read_csv("./voted_gt/ground_truth.csv")
    # inner join result_df and voted_truth
    voted_result = result_df.merge(voted_truth, on="node_id", how="inner")

    # count the percentage of agreement on "voted_gt" and "label"
    voted_result["agreement"] = voted_result.apply(lambda x: 1 if x["label"] == x["voted_gt"] else 0, axis=1)
    # add up the "agreement" column
    sum = voted_result["agreement"].sum()
    print("Provisional test accuracy:", sum / len(voted_result))
