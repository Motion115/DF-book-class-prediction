import os
from dataloader import DataPreprocessor
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.GCN import GCN
import numpy as np
from copy import deepcopy
import pandas as pd

if __name__ == "__main__":
    root = "./data"
    weights_directory = "./weights/" + "20231026-164707"

    data_path = os.path.join(root, 'Children.csv')
    # data_preprocessor = DataPreprocessor(data_path, SequenceEncoder())
    data_preprocessor = DataPreprocessor(data_path, load_feature_from_disk="./data/node_attr.pt")

    data = data_preprocessor.graph

    num_node_features=data.x.shape[1]
    model= GCN(num_node_features, num_classes=24)

    # load checkpoint
    checkpoint = torch.load(os.path.join(weights_directory, "990_model.pt"))
    # load best model parameter
    model.load_state_dict(checkpoint)

    model.eval()
    _, pred = model(data).max(dim=1)
 
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
