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

class BookDataset(Dataset):
    def __init__(self, data, random_seed=42):
        super(BookDataset, self).__init__()
        # stable randon seed, so that the split is the same for all runs
        random.seed(random_seed)
        # shuffle
        random.shuffle(data)
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return [self.data[idx]["anchor"], self.data[idx]["positive"], self.data[idx]["negative"]]

def train(device, train_loader, net, optimizer, criterion):
    train_loss = 0.0
    for i, data in tqdm(enumerate(train_loader, 0), desc="iters"):
        anchor, positive, negative = data[0], data[1], data[2]
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        anchor_to_embedding = net(anchor)
        positive_to_embedding = net(positive)
        negative_to_embedding = net(negative)
        
        optimizer.zero_grad()
        loss = criterion(anchor_to_embedding, positive_to_embedding, negative_to_embedding)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    loss = train_loss / len(train_loader)
    return loss

if __name__ == '__main__':
    data_preprocessor, model = load_data(
        embedding_filename="./data/bert-cls-embeddings.pth",
        model_class=MLP,
        sampling_strategy=None,
    )
    data = data_preprocessor.graph
    vectors = data.x
    labels = data.y

    # group vectors by labels
    grouped_vectors = {}
    for i in trange(len(labels)):
        label = int(labels[i])
        if label not in grouped_vectors:
            grouped_vectors[label] = []
        grouped_vectors[label].append(vectors[i])
    
    # remove the key -1
    del grouped_vectors[-1]

    # the classes span from 0 - 23
    # the entries spans from 61 - 19647

    UPSAMPLE_COUNT = 20000
    triplet_dataset = []

    # generate a list with range 0 - 23
    candidancy_list = list(range(0, 24))

    for key, vector_list in tqdm(grouped_vectors.items()):
        # remove key in class_candidancy_list
        class_candidancy_list = candidancy_list.copy()
        del class_candidancy_list[class_candidancy_list.index(key)]
        # print(class_candidancy_list)
        
        # length of vector_list
        vector_list_length = len(vector_list)
        inclass_candidancy_list = list(range(0, vector_list_length))
        for i in range(UPSAMPLE_COUNT):
            anchor_idx = i % vector_list_length
            real_inclass_candidancy_list = inclass_candidancy_list.copy()
            del real_inclass_candidancy_list[real_inclass_candidancy_list.index(anchor_idx)]
            positive_idx = random.choice(real_inclass_candidancy_list)
            
            negative_class_idx = random.choice(class_candidancy_list)
            negative_vector = random.choice(grouped_vectors[negative_class_idx])
            # anchor, positive, negative
            triplet_dataset.append({
                "anchor": vector_list[anchor_idx], 
                "positive": vector_list[positive_idx],
                "negative": negative_vector})
    
    print(len(triplet_dataset))
    triplet_size = sys.getsizeof(triplet_dataset)
    print("Size of triplet_dataset:", triplet_size, "bytes")

    current_time = time.strftime("%Y%m%d-%H%M%S")
    weights_directory = "./fuse_weight/" + current_time

    # start the training process
    EPOCHS = 200

    dataset = BookDataset(triplet_dataset)
    dataloader = DataLoader(dataset, batch_size=10384, shuffle=True)

    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.TripletMarginLoss()

    net = Fusion().to(device)
    # use adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-08)
    
    bench_loss = 1000
    for epoch in range(0, EPOCHS):
        loss = train(device, dataloader, net, optimizer, criterion)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.flush()

        print('epoch:{}, loss:{}'.format(epoch + 1, loss * 100))
        print("------------------------------------")
        # only store the models that imporve on validation and drop in loss
        if loss < bench_loss or epoch % 10 == 0 :
            bench_loss = loss

            print('Saving model...')
            if not os.path.isdir(weights_directory):
                os.mkdir(weights_directory)
            torch.save(net.state_dict(), weights_directory + '/fuse_epoch_{}.ckpt'.format(epoch+1))

    print('Finished Training!')





