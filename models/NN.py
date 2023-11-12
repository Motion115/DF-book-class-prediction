import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(MLP, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(num_node_features, 60),
            nn.ReLU(),
            nn.Linear(60, 35),
            nn.ReLU(),
            nn.Linear(35, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fusion(x)
        x = self.softmax(x)
        return x