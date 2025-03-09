import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
import time

# Define a basic residual block for ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Implements a residual block with two convolutional layers.
        - Each convolution uses a 3x3 kernel, stride=1 (except when downsampling).
        - Includes BatchNorm and ReLU activation.
        - If the input and output dimensions do not match, a 1x1 convolution is used.

        Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - stride (int): Stride for the first convolutional layer.
        """
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (identity mapping)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # Apply shortcut transformation if necessary
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out

# Define ResNet-18 Model
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        """
        Implements ResNet-18 architecture with:
        - Initial convolutional layer
        - Four groups of residual blocks
        - Global average pooling
        - Fully connected classification layer
        """
        super(ResNet18, self).__init__()

        # Initial convolutional layer (3 input channels -> 64 output channels)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Four groups of residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Creates a sequence of residual blocks."""
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Defines forward pass."""
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

# Main training function
def train(args):
    """Trains the ResNet-18 model on CIFAR-10."""
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Define data transformations
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Random cropping with padding
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=args.num_workers)

    # Initialize model, loss function, and optimizer
    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Select optimizer
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

        # C2 - Measure Data Loading Time
        data_load_start = time.perf_counter()
        data = list(train_loader)
        data_load_end = time.perf_counter()

        # C2 - Training Time
        train_start = time.perf_counter()
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training loss and accuracy
            epoch_loss += loss.item()
            batch_accuracy = compute_accuracy(outputs, targets)
            epoch_correct += batch_accuracy * targets.size(0)
            total_samples += targets.size(0)

            # Print per-batch loss and accuracy
            print(f"Epoch [{epoch+1}/5], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}")

        print(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {epoch_correct/total_samples:.4f}")

        train_end = time.perf_counter()

        # C2 - Timing Metrics
        data_load_time = data_load_end - data_load_start
        train_time = train_end - train_start
        total_time = train_time + data_load_time
        print(f"Epoch {epoch+1} | Data Load Time: {data_load_time:.4f}s | Train Time: {train_time:.4f}s | Total Time: {total_time:.4f}s")

# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10")
    
    # C1
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to CIFAR-10 dataset")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer to use: sgd, adam, adagrad, etc.")
    args = parser.parse_args()
    train(args)
