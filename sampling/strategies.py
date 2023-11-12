import torch
import random

class SamplingStrategies():
    def __init__(self, original_data, train_ids) -> None:
        # original_data is a panda series, train_ids is a torch tensor
        self.train_ids = train_ids
        self.data = original_data[["node_id", "label"]]
        self.category_data, self.min_samples, self.max_samples = self._categorization()
    
    def _categorization(self):
        categoies_dict = {}
        for idx in self.train_ids:
            idx = int(idx)
            if self.data.iloc[idx]["label"] not in categoies_dict:
                categoies_dict[self.data.iloc[idx]["label"]] = [int(self.data.iloc[idx]["node_id"])]
            else:
                categoies_dict[self.data.iloc[idx]["label"]].append(int(self.data.iloc[idx]["node_id"]))
        min_samples = min([len(v) for _, v in categoies_dict.items()])
        max_samples = max([len(v) for _, v in categoies_dict.items()])
        return categoies_dict, min_samples, max_samples

    def downsample_balancing(self, options=None):
        # Downsample to even the class bias, applicable to all trainings
        # options: TBD (E.g. how to select representative samples, default to random)
        examples = []
        for key, value in self.category_data.items():
            # for each value, randomly sample self.min_samples, without repetition
            examples.extend(random.sample(value, self.min_samples))
        # examples will be the train_ids, the rest will be val_ids
        train_ids = examples
        # val_ids = [int(i) for i in self.train_ids if i not in train_ids]
        return train_ids, self.train_ids


    def upsample_balancing(self):
        """
        Upsample the data to even class bias, only support Non-Graph methods
        """
        pass