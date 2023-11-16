import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
# from preprocessing import prepare_dataset
# from torch.utils.data import Dataset
# from tqdm import tqdm

# from dgl.dataloading import GraphDataLoader


"""
    directional_GSN
    directional_GSN is a combination of Graph Substructure Networks (GSN) with Directional Graph Networks (DGN), where we defined a vector field based on substructure encoding instead of Laplacian eigenvectors.
"""

def aggregate_mean(h, vector_field, h_in):
    return torch.mean(h, dim=1)


def aggregate_max(h, vector_field, h_in):
    return torch.max(h, dim=1)[0]


def aggregate_sum(h, vector_field, h_in):
    return torch.sum(h, dim=1)


def aggregate_dir_dx(h, vector_field, h_in, eig_idx=1):
    eig_w = (
        (vector_field[:, :, eig_idx])
        / (
            torch.sum(
                torch.abs(vector_field[:, :, eig_idx]), keepdim=True, dim=1
            )
            + 1e-8
        )
    ).unsqueeze(-1)
    h_mod = torch.mul(h, eig_w)
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in)


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(FCLayer, self).__init__()
        
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, 1 / self.in_size)
        self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        return h


class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.fc = FCLayer(in_size, out_size)

    def forward(self, x):
        x = self.fc(x)
        return x


class DGNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, aggregators):
        super().__init__()
        self.dropout = dropout
        self.aggregators = aggregators
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.pretrans = MLP(in_size=2 * in_dim, out_size=in_dim)
        self.posttrans = MLP(
            in_size=(len(aggregators) * 1 + 1) * in_dim, out_size=out_dim
        )

    def pretrans_edges(self, edges):
        z2 = torch.cat([edges.src["h"], edges.dst["h"]], dim=1)
        vector_field = edges.data["eig"]
        return {"e": self.pretrans(z2), "vector_field": vector_field}

    def message_func(self, edges):
        return {
            "e": edges.data["e"],
            "vector_field": edges.data["vector_field"],
        }

    def reduce_func(self, nodes):
        h_in = nodes.data["h"]
        h = nodes.mailbox["e"]

        vector_field = nodes.mailbox["vector_field"]

        h = torch.cat(
            [
                aggregate(h, vector_field, h_in)
                for aggregate in self.aggregators
            ],
            dim=1,
        )

        return {"h": h}

    def forward(self, g, h, snorm_n):
        g.ndata["h"] = h

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata["h"]], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization
        h = h * snorm_n
        h = self.batchnorm_h(h)
        h = F.relu(h)

        h = F.dropout(h, self.dropout, training=self.training)

        return h


class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2**l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2**L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GraphSubstructureModel(nn.Module):
    def __init__(self, hidden_dim=420, out_dim=420, dropout=0.2, n_layers=4):
        super().__init__()
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)
        self.aggregators = [
            aggregate_mean,
            aggregate_sum,
            aggregate_max,
            aggregate_dir_dx,
        ]

        self.layers = nn.ModuleList(
            [
                DGNLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    dropout=dropout,
                    aggregators=self.aggregators,
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.layers.append(
            DGNLayer(
                in_dim=hidden_dim,
                out_dim=out_dim,
                dropout=dropout,
                aggregators=self.aggregators,
            )
        )

        # 128 out dim since ogbg-molpcba has 128 tasks
        self.MLP_layer = MLPReadout(out_dim, 128)

    def forward(self, g, h, snorm_n):
        h = self.embedding_h(h)
        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, snorm_n)
            h = h_t

        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")

        return self.MLP_layer(hg)

    def loss(self, scores, labels):
        is_labeled = labels == labels
        loss = nn.BCEWithLogitsLoss()(
            scores[is_labeled], labels[is_labeled].float()
        )
        return loss