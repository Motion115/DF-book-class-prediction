import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels=num_node_features,
                                    out_channels=16,
                                    heads=2)
        self.conv2 = GATConv(in_channels=2*16,
                                    out_channels=num_classes,
                                    heads=1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)