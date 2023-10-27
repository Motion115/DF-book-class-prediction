import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from models.GCN import GCN
from utils import load_data

if __name__ == '__main__':
    # get current time
    current_time = time.strftime("%Y%m%d-%H%M%S")
    weights_directory = "./weights/" + current_time
    # create weights directory
    if not os.path.exists(weights_directory):
        os.makedirs(weights_directory)

    data_preprocessor, model = load_data("./data/bert-cls-embeddings.pth", GCN)
    data = data_preprocessor.graph

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    # judge if cuda is present
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model and data to device
    model.to(device)
    data = data.to(device)

    # initialize summary writer
    writer = SummaryWriter()

    num_epochs = 800
    loss_minimum = 1000

    model.train()
    # train the model
    with tqdm(total=num_epochs) as t:
        for epoch in range(num_epochs):

            out = model(data)
            optimizer.zero_grad()
            loss = loss_function(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            t.set_description("Epoch %i" % epoch)
            t.set_postfix(loss=loss.item())
            t.update(1)

            # save the model with the lowest loss
            if loss.item() < loss_minimum:
                loss_minimum = loss.item()
                torch.save(model.state_dict(), os.path.join(weights_directory, str(epoch + 1) + "_model.pt"))
            
            # Add loss value to TensorBoard
            writer.add_scalar("Loss/train", loss.item(), epoch)
        

    model.eval()
    _, pred = model(data).max(dim=1)

    correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    acc = correct / int(data.train_mask.sum())
    print('GCN Accuracy on train set: {:.4f}'.format(acc))