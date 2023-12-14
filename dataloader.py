import os
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
# from encoding.sentence_transformer import SentenceTransformer
from torch_geometric.data import Data
from sampling.strategies import SamplingStrategies

class DataPreprocessor():
    def __init__(self, raw_data_source, feat_eng_method=None, load_feature_from_disk=None, sampling_strategy=None) -> None:
        self.raw_data_source = pd.read_csv(raw_data_source)
        self.num_nodes = self.raw_data_source.node_id.nunique()
        COO_mat = self.build_COO_matrix(src_col='node_id', dst_col='neighbour')
        if load_feature_from_disk is not None:
            node_attributes = torch.load(load_feature_from_disk)
            # transform node_attributes into a torch tensor
            node_attributes = torch.tensor(node_attributes, dtype=torch.float)
            # print(node_attributes.shape)
        elif load_feature_from_disk is not None:
            node_attributes = self.build_node_attribute(attr_col="text", feat_eng_method=feat_eng_method)
        else:
            node_attributes = None
        
        labels = self.raw_data_source["label"]
        # replace all empty values with -1
        labels = labels.replace(np.nan, -1)
        # set all values to integer
        labels = labels.astype(int)
        class_id = torch.tensor(labels, dtype=torch.long)
        train_mask, val_mask, test_mask = self.build_mask(judge_col="label", attr_col="node_id", sampling_strategy=sampling_strategy)
        self.graph = Data(x=node_attributes, edge_index=COO_mat, y=class_id, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        self.masks = {
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask":test_mask
        }
    
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
    
    def build_mask(self, judge_col, attr_col, sampling_strategy):
        # check if "label" col have value, if true, then train set
        # is the data that have label value, otherwise, test set
        if judge_col in self.raw_data_source.columns:
            train_ids = self.raw_data_source[self.raw_data_source[judge_col].notnull()][attr_col].values
            test_ids = self.raw_data_source[self.raw_data_source[judge_col].isnull()][attr_col].values
        else:
            raise ValueError("judge_col is not in the dataframe")
        
        # generate train_mask and test_mask
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)


        if sampling_strategy == None:
            # randomly split train_ids 8:2
            train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=31)
        elif sampling_strategy == "downsample":
            sampling = SamplingStrategies(
            original_data=self.raw_data_source,
            train_ids=train_ids,
            )
            train_ids, val_ids = sampling.downsample_balancing()

        train_mask[torch.tensor(train_ids)] = True
        val_mask[torch.tensor(val_ids)] = True
        test_mask[torch.tensor(test_ids)] = True
        return train_mask, val_mask, test_mask

if __name__ == '__main__':
    root = "./data"
    data_path = os.path.join(root, 'Children.csv')
    # data_preprocessor = DataPreprocessor(data_path, SequenceEncoder())
    data_preprocessor = DataPreprocessor(data_path, sampling_strategy="downsample")