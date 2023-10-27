import os
import ast
import pandas as pd
import torch
import numpy as np
from encoding.sentence_transformer import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

class DataPreprocessor():
    def __init__(self, raw_data_source, feat_eng_method=None, load_feature_from_disk=None) -> None:
        self.raw_data_source = pd.read_csv(raw_data_source)
        self.num_nodes = self.raw_data_source.node_id.nunique()
        COO_mat = self.build_COO_matrix(src_col='node_id', dst_col='neighbour')
        if load_feature_from_disk is not None:
            node_attributes = torch.load(load_feature_from_disk)
            # transform node_attributes into a torch tensor
            node_attributes = torch.tensor(node_attributes, dtype=torch.float)
            # print(node_attributes.shape)
        else:
            node_attributes = self.build_node_attribute(attr_col="text", feat_eng_method=feat_eng_method)
        
        labels = self.raw_data_source["label"]
        # replace all empty values with -1
        labels = labels.replace(np.nan, -1)
        # set all values to integer
        labels = labels.astype(int)
        class_id = torch.tensor(labels, dtype=torch.long)
        train_mask, test_mask = self.build_mask(judge_col="label", attr_col="node_id")
        self.graph = Data(x=node_attributes, edge_index=COO_mat, y=class_id, train_mask=train_mask, test_mask=test_mask)
    
    def build_COO_matrix(self, src_col, dst_col):
        COO_mat = []
        # traverse through the raw_data_source
        for row, col in zip(self.raw_data_source[src_col], self.raw_data_source[dst_col]):
            # parse the string in col into a list
            col_list = ast.literal_eval(col)
            for col_item in col_list:
                COO_mat.append([int(row), int(col_item)])
        # COO_mat to tensor
        COO_mat = torch.tensor(COO_mat, dtype=torch.long).T
        return COO_mat

    def build_node_attribute(self, attr_col, feat_eng_method):
        node_attr = feat_eng_method(self.raw_data_source[attr_col])
        torch.save(node_attr, "./data/node_attr.pt")
        return node_attr
    
    def build_mask(self, judge_col, attr_col):
        # check if "label" col have value, if true, then train set
        # is the data that have label value, otherwise, test set
        if judge_col in self.raw_data_source.columns:
            train_ids = self.raw_data_source[self.raw_data_source[judge_col].notnull()][attr_col].values
            test_ids = self.raw_data_source[self.raw_data_source[judge_col].isnull()][attr_col].values
        else:
            raise ValueError("judge_col is not in the dataframe")

        # generate train_mask and test_mask
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        train_mask[torch.tensor(train_ids)] = True
        test_mask[torch.tensor(test_ids)] = True

        return train_mask, test_mask

if __name__ == '__main__':
    root = "./data"
    data_path = os.path.join(root, 'Children.csv')
    # data_preprocessor = DataPreprocessor(data_path, SequenceEncoder())
    data_preprocessor = DataPreprocessor(data_path, load_feature_from_disk="./data/node_attr.pt")