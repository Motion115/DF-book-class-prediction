import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import time
import random
from models.NN import MLP
from utils import load_data

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(768, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 150)
        )

    def forward(self, x):
        x = self.fusion(x)
        return x


if __name__ == '__main__':
    data_preprocessor, model = load_data(
        embedding_filename="./data/bert-cls-embeddings.pth",
        model_class=MLP,
        sampling_strategy=None,
    )
    data = data_preprocessor.graph
    vectors = data.x
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data.to(device)
    net = Fusion().to(device)
    # load weights
    net.load_state_dict(torch.load("./fuse_weight/20231207-113350/fuse_epoch_51.ckpt"))
    # inference from original vectors to encoded vectors
    net.eval()
    with torch.no_grad():
        res_vectors = net(vectors.to(device))

    res_vector = res_vectors.clone().detach().cpu()
    # save res_vectors as fuse-embeddings.pth
    torch.save(res_vector, "./data/fuse-embeddings.pth")