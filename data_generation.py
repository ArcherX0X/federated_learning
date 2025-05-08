import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from collections import defaultdict
import random

# Setup
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
label_indices = defaultdict(list)
for idx, (_, label) in enumerate(mnist_data):
    label_indices[label].append(idx)

# Shuffle indices
for indices in label_indices.values():
    random.shuffle(indices)

# Create output directories
base_path = "experiments"
os.makedirs(base_path, exist_ok=True)


# Helper function to save client indices
def save_client_indices(dataset_name, client_data):
    path = os.path.join(base_path, dataset_name)
    os.makedirs(path, exist_ok=True)
    for i, indices in enumerate(client_data):
        torch.save(indices, os.path.join(path, f"client_{i}_indices.pt"))


# --- BASELINE: 10 clients, 2k each, uniform label (10% per label), no overlap ---
baseline_data = []
label_quota = 200  # 10% of 2k
label_copy = {k: v.copy() for k, v in label_indices.items()}
for _ in range(10):
    client_indices = []
    for lbl in range(10):
        client_indices.extend(label_copy[lbl][:label_quota])
        label_copy[lbl] = label_copy[lbl][label_quota:]
    baseline_data.append(client_indices)
save_client_indices("baseline", baseline_data)

# --- EXPERIMENT 1: Varying size (500 to 3500), uniform labels, 10% per label ---
exp1_data = []
sizes = [
    500,
    750,
    1000,
    1250,
    1500,
    1750,
    2000,
    2250,
    2500,
    2750,
]  # 500 to 2750, 250 increments
label_copy = {k: v.copy() for k, v in label_indices.items()}
for size in sizes:
    per_label = int(size * 0.1)
    client_indices = []
    for lbl in range(10):
        client_indices.extend(label_copy[lbl][:per_label])
        label_copy[lbl] = label_copy[lbl][per_label:]
    exp1_data.append(client_indices)
save_client_indices("experiment1_uniform_size", exp1_data)

# --- EXPERIMENT 2: Fixed size (2k), skewed label, no overlap, all labels, 2%-60% ---
exp2_data = []
total_clients = 10
client_size = 2000
label_copy = {k: v.copy() for k, v in label_indices.items()}


def generate_skewed_label_dist():
    while True:
        proportions = np.random.dirichlet(np.ones(10), 1).flatten()
        percentages = (proportions * 100).round(2)
        if all(2 <= p <= 60 for p in percentages):
            return (proportions * client_size).astype(int)


for _ in range(total_clients):
    label_counts = generate_skewed_label_dist()
    client_indices = []
    for lbl, count in enumerate(label_counts):
        client_indices.extend(label_copy[lbl][:count])
        label_copy[lbl] = label_copy[lbl][count:]
    exp2_data.append(client_indices)
save_client_indices("experiment2_skewed_labels", exp2_data)

# --- EXPERIMENT 3: Combine Exp1 + Exp2 (skewed label + variable sizes) ---
exp3_data = []
label_copy = {k: v.copy() for k, v in label_indices.items()}
for size in sizes:
    proportions = generate_skewed_label_dist()
    proportions = proportions / proportions.sum()  # normalize to sum to 1
    label_counts = (proportions * size).astype(int)
    client_indices = []
    for lbl, count in enumerate(label_counts):
        client_indices.extend(label_copy[lbl][:count])
        label_copy[lbl] = label_copy[lbl][count:]
    exp3_data.append(client_indices)
save_client_indices("experiment3_skewed_labels_and_size", exp3_data)
