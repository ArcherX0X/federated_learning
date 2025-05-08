from fed_model import MLP, train_local, evaluate, fed_avg_weighted
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os


def load_client_datasets(path):
    transform = transforms.ToTensor()
    full_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    client_datasets = []
    sizes = []
    for i in range(10):
        idx_path = os.path.join(path, f"client_{i}_indices.pt")
        indices = torch.load(idx_path)
        client_datasets.append(Subset(full_data, indices))
        sizes.append(len(indices))
    return client_datasets, sizes


def run_federated_learning(dataset_path, rounds=50):
    print(f"\n🔁 Training on: {dataset_path}")
    client_datasets, sizes = load_client_datasets(dataset_path)
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    global_model = MLP()
    criterion = torch.nn.CrossEntropyLoss()

    for r in range(rounds):
        local_models = []
        for dataset in client_datasets:
            local_model = MLP()
            local_model.load_state_dict(global_model.state_dict())
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
            train_local(local_model, loader, criterion, optimizer, epochs=1)
            local_models.append(local_model)

        global_model = fed_avg_weighted(local_models, sizes)
        acc = evaluate(global_model, test_loader)
        print(f"Round {r+1:02d} - Global Accuracy: {acc:.4f}")


# Run experiments
base_path = "experiments"
for dname in [
    "baseline",
    "experiment1_uniform_size",
    "experiment2_skewed_labels",
    "experiment3_skewed_labels_and_size",
]:
    run_federated_learning(os.path.join(base_path, dname))
