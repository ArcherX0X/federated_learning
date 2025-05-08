from fed_model import MLP, train_local, evaluate, fed_avg_weighted
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np


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


def evaluate_loss(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in dataloader:
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
    return total_loss / total_samples


def compute_confusion_matrix(model, dataloader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            preds = model(x).argmax(dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())
    return confusion_matrix(y_true, y_pred)


def run_federated_learning(dataset_path, dataset_name, rounds=50):
    print(f"\n🔁 Training on: {dataset_name}")
    client_datasets, sizes = load_client_datasets(dataset_path)
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    global_model = MLP()
    criterion = torch.nn.CrossEntropyLoss()
    results = []

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
        loss = evaluate_loss(global_model, test_loader, criterion)
        print(f"Round {r+1:02d} - Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        results.append({"Round": r + 1, "Accuracy": acc, "Loss": loss})

    # Save accuracy + loss log
    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(f"results/{dataset_name}_accuracy.csv", index=False)
    print(f"✅ Saved results to results/{dataset_name}_accuracy.csv")

    # Final confusion matrix
    cm = compute_confusion_matrix(global_model, test_loader)
    np.savetxt(
        f"results/{dataset_name}_confusion_matrix.csv", cm, delimiter=",", fmt="%d"
    )
    print(f"✅ Saved confusion matrix to results/{dataset_name}_confusion_matrix.csv")
    print(f"\n🔁 Training on: {dataset_name}")
    client_datasets, sizes = load_client_datasets(dataset_path)
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    global_model = MLP()
    criterion = torch.nn.CrossEntropyLoss()
    results = []

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
        print(f"Round {r+1:02d} - Accuracy: {acc:.4f}")
        results.append({"Round": r + 1, "Accuracy": acc})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(f"results/{dataset_name}_accuracy.csv", index=False)
    print(f"✅ Saved results to results/{dataset_name}_accuracy.csv")

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
if __name__ == "__main__":
    datasets_to_run = [
        ("baseline", "baseline"),
        ("experiment1_uniform_size", "exp1_uniform_size"),
        ("experiment2_skewed_labels", "exp2_skewed_labels"),
        ("experiment3_skewed_labels_and_size", "exp3_skewed_labels_and_size"),
    ]

    for folder, label in datasets_to_run:
        path = os.path.join("experiments", folder)
        run_federated_learning(path, label)
