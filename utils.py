import time
import os
from dataloader import DataPreprocessor

def load_data(embedding_filename, model_class, sampling_strategy=None):
    if embedding_filename == "":
        raise ValueError("embedding_filename is empty")
    root = "./data"
    data_path = os.path.join(root, 'Children.csv')
    data_preprocessor = DataPreprocessor(
        raw_data_source=data_path,
        load_feature_from_disk=embedding_filename,
        sampling_strategy=sampling_strategy
        )
    data = data_preprocessor.graph

    num_node_features=data.x.shape[1]
    model= model_class(num_node_features, num_classes=24)
    return data_preprocessor, model 