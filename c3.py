import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define a basic residual block for ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# Define ResNet-18 Model
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Function to compute training accuracy
def compute_accuracy(outputs, targets):
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    return correct / targets.size(0)

# Main training function (C1 + C2)
def train(args):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=args.num_workers)

    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    elif args.optimizer == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=0.1, weight_decay=5e-4)
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=0.1, weight_decay=5e-4)
    else:
        raise ValueError("Unsupported optimizer! Choose from: sgd, adam, adagrad, adadelta.")

    # Training loop for 5 epochs
    for epoch in range(5):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        total_samples = 0

        # C2: Measure Data Loading Time
        data_load_start = time.perf_counter()
        data = list(train_loader)  # Trigger data loading for the epoch
        data_load_end = time.perf_counter()

        # C2: Measure Training Time
        train_start = time.perf_counter()
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_accuracy = compute_accuracy(outputs, targets)
            epoch_correct += batch_accuracy * targets.size(0)
            total_samples += targets.size(0)

            print(f"Epoch [{epoch+1}/5], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}")

        print(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {epoch_correct/total_samples:.4f}")

        train_end = time.perf_counter()
        data_load_time = data_load_end - data_load_start
        train_time = train_end - train_start
        total_time = data_load_time + train_time
        print(f"Epoch {epoch+1} | Data Load Time: {data_load_time:.4f}s | Train Time: {train_time:.4f}s | Total Time: {total_time:.4f}s")

# C3: I/O Optimization Experiment in the original code base
def run_io_experiment(args, num_workers_list):
    io_times = []
    # Use the same data transformation to ensure consistency of the experiment
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    # Test for each specified num_workers
    for workers in num_workers_list:
        print(f"\nTesting with num_workers = {workers}")
        args.num_workers = workers
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=args.num_workers)

        epoch_io_times = []
        # Run 3 epochs for stable average measurement of I/O loading time
        for epoch in range(3):
            start_time = time.perf_counter()
            # Iterate through all batches to trigger data loading without training
            for _ in train_loader:
                pass
            end_time = time.perf_counter()
            epoch_io_time = end_time - start_time
            epoch_io_times.append(epoch_io_time)
            print(f"Epoch {epoch+1} I/O Time: {epoch_io_time:.4f}s")
        avg_io_time = sum(epoch_io_times) / len(epoch_io_times)
        io_times.append(avg_io_time)
        print(f"Average I/O Time for num_workers {workers}: {avg_io_time:.4f}s")
    
    # Draw the result graph
    plt.figure(figsize=(8, 5))
    plt.plot(num_workers_list, io_times, marker='o')
    plt.xlabel("Number of DataLoader Workers")
    plt.ylabel("Average Data Loading Time (s)")
    plt.title("I/O Time vs. Number of Workers")
    plt.grid(True)
    plt.savefig("io_time.png") 
    best_workers = num_workers_list[io_times.index(min(io_times))]
    print(f"Best performance achieved with num_workers = {best_workers}")

# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10 and perform I/O optimization experiments")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to CIFAR-10 dataset")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer to use: sgd, adam, adagrad, adadelta")
    # New argument to control whether to run the I/O optimization experiment (C3)
    parser.add_argument("--experiment3", action="store_true", help="Run I/O optimization experiment for C3")
    args = parser.parse_args()

    if args.experiment3:
        # Test with different numbers of workers
        workers_to_test = [0, 4, 8, 12, 16]
        run_io_experiment(args, workers_to_test)
    else:
        train(args)
