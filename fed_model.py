import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import os
import copy


# Simple MLP Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Local Training
def train_local(model, dataloader, criterion, optimizer, epochs=1):
    model.train()
    for _ in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()


# Evaluation
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# Weighted FedAvg (based on number of samples)
def fed_avg_weighted(local_models, local_sizes):
    global_model = copy.deepcopy(local_models[0])
    total = sum(local_sizes)
    for key in global_model.state_dict().keys():
        weighted_sum = sum(
            model.state_dict()[key] * (sz / total)
            for model, sz in zip(local_models, local_sizes)
        )
        global_model.state_dict()[key].copy_(weighted_sum)
    return global_model
