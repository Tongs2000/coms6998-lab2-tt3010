import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
import time
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend for plotting
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# BasicBlock WITH BatchNorm (existing)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# BasicBlock WITHOUT BatchNorm (for C7)
# ----------------------------------------------------------------------
class BasicBlockNoBN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # No BatchNorm in the shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

# ----------------------------------------------------------------------
# ResNet-18 WITH BatchNorm (existing)
# ----------------------------------------------------------------------
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
            layers.append(BasicBlock(out_channels, out_channels))
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

# ----------------------------------------------------------------------
# ResNet-18 WITHOUT BatchNorm (for C7)
# ----------------------------------------------------------------------
class ResNet18NoBN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18NoBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlockNoBN(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlockNoBN(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# ----------------------------------------------------------------------
# Function to compute training accuracy
# ----------------------------------------------------------------------
def compute_accuracy(outputs, targets):
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    return correct / targets.size(0)

# ----------------------------------------------------------------------
# Create an optimizer with consistent hyperparams
# ----------------------------------------------------------------------
def create_optimizer(optimizer_name, params):
    """
    Creates an optimizer using the given name and parameters,
    with default hyperparameters specified in the assignment:
    - learning rate = 0.1
    - weight decay = 5e-4
    - momentum = 0.9 (when it applies)
    - nesterov for "sgd_nesterov"
    """
    lr = 0.1
    wd = 5e-4
    momentum = 0.9

    if optimizer_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    elif optimizer_name == "sgd_nesterov":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    elif optimizer_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif optimizer_name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=wd)
    elif optimizer_name == "adadelta":
        return optim.Adadelta(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# ----------------------------------------------------------------------
# Original train() function
# ----------------------------------------------------------------------
def train(args):
    """
    Trains the original ResNet18 with BatchNorm for 5 epochs
    using the chosen optimizer (from create_optimizer).
    """
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
    optimizer = create_optimizer(args.optimizer, model.parameters())

    for epoch in range(5):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        total_samples = 0

        # Synchronize before timing data loading
        if device.type == "cuda":
            torch.cuda.synchronize()
        data_load_start = time.perf_counter()

        data = list(train_loader)  # Trigger data loading for the epoch

        if device.type == "cuda":
            torch.cuda.synchronize()
        data_load_end = time.perf_counter()

        # Synchronize before training
        if device.type == "cuda":
            torch.cuda.synchronize()
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

            print(f"Epoch [{epoch+1}/5], Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_end = time.perf_counter()

        data_load_time = data_load_end - data_load_start
        train_time = train_end - train_start
        total_time = data_load_time + train_time

        print(f"Epoch {epoch+1} completed. "
              f"Avg Loss: {epoch_loss/len(train_loader):.4f}, "
              f"Accuracy: {epoch_correct/total_samples:.4f}")
        print(f"Epoch {epoch+1} | Data Load Time: {data_load_time:.4f}s | "
              f"Train Time: {train_time:.4f}s | Total Time: {total_time:.4f}s")
# ----------------------------------------------------------------------
# Helper to create an optimizer with consistent hyperparams
# ----------------------------------------------------------------------
def create_optimizer(optimizer_name, params):
    """
    Creates an optimizer using the given name and parameters,
    with default hyperparameters specified in the assignment:
    - learning rate = 0.1
    - weight decay = 5e-4
    - momentum = 0.9 (when it applies)
    - nesterov for "sgd_nesterov"
    """
    lr = 0.1
    wd = 5e-4
    momentum = 0.9

    if optimizer_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    elif optimizer_name == "sgd_nesterov":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    elif optimizer_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif optimizer_name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=wd)
    elif optimizer_name == "adadelta":
        return optim.Adadelta(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# ----------------------------------------------------------------------
# I/O experiment (C3) to find best num_workers
# ----------------------------------------------------------------------
def run_io_experiment(args, num_workers_list):
    io_times = []
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    for workers in num_workers_list:
        print(f"\nTesting with num_workers = {workers}")
        args.num_workers = workers
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=args.num_workers)

        epoch_io_times = []
        for epoch in range(3):
            if torch.cuda.is_available() and args.cuda:
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in train_loader:
                pass

            if torch.cuda.is_available() and args.cuda:
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            epoch_io_time = end_time - start_time
            epoch_io_times.append(epoch_io_time)
            print(f"Epoch {epoch+1} I/O Time: {epoch_io_time:.4f}s")
        avg_io_time = sum(epoch_io_times) / len(epoch_io_times)
        io_times.append(avg_io_time)
        print(f"Average I/O Time for num_workers {workers}: {avg_io_time:.4f}s")

    plt.figure(figsize=(8, 5))
    plt.plot(num_workers_list, io_times, marker='o')
    plt.xlabel("Number of DataLoader Workers")
    plt.ylabel("Average Data Loading Time (s)")
    plt.title("I/O Time vs. Number of Workers")
    plt.grid(True)
    plt.savefig("io_time.png")

    best_workers = num_workers_list[io_times.index(min(io_times))]
    print(f"Best performance achieved with num_workers = {best_workers}")
    return best_workers

# ----------------------------------------------------------------------
# C4: measure loading and compute time for a single epoch
# ----------------------------------------------------------------------
def measure_loading_and_computation(args, workers):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    args.num_workers = workers

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=workers)

    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args.optimizer, model.parameters())

    if device.type == "cuda":
        torch.cuda.synchronize()
    data_load_start = time.perf_counter()
    data = list(train_loader)
    if device.type == "cuda":
        torch.cuda.synchronize()
    data_load_end = time.perf_counter()
    data_load_time = data_load_end - data_load_start

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_start = time.perf_counter()
    for inputs, targets in data:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_end = time.perf_counter()
    compute_time = train_end - train_start

    return data_load_time, compute_time

# ----------------------------------------------------------------------
# C4 experiment
# ----------------------------------------------------------------------
def run_c4_experiment(args):
    workers_to_test = [0, 4, 8, 12, 16]
    best_workers = run_io_experiment(args, workers_to_test)

    print("\n[C4] Profiling with num_workers = 1")
    load_time_1, compute_time_1 = measure_loading_and_computation(args, 1)
    print(f"num_workers=1 -> Data Load Time: {load_time_1:.4f}s, Compute Time: {compute_time_1:.4f}s")

    print(f"\n[C4] Profiling with num_workers = {best_workers}")
    load_time_best, compute_time_best = measure_loading_and_computation(args, best_workers)
    print(f"num_workers={best_workers} -> Data Load Time: {load_time_best:.4f}s, Compute Time: {compute_time_best:.4f}s")

    total_1 = load_time_1 + compute_time_1
    total_best = load_time_best + compute_time_best
    print("\n[C4] Comparison of 1 worker vs. best workers:")
    print(f"1 worker      => Data Load: {load_time_1:.4f}s, Compute: {compute_time_1:.4f}s, Total: {total_1:.4f}s")
    print(f"{best_workers} workers => Data Load: {load_time_best:.4f}s, Compute: {compute_time_best:.4f}s, Total: {total_best:.4f}s")

# ----------------------------------------------------------------------
# C6 experiment: different optimizers
# ----------------------------------------------------------------------
def run_c6_experiment(args):
    """
    1. Finds best num_workers from C3 (or uses a known best if desired).
    2. For each optimizer (SGD, SGD with Nesterov, Adagrad, Adadelta, Adam),
       runs 5 epochs on GPU (if available) and measures:
         - Average training time per epoch
         - Final training loss (5th epoch)
         - Final training accuracy (5th epoch)
    3. Prints out a comparison and brief discussion.
    """
    # Step 1: find the best num_workers from C3
    workers_to_test = [0, 4, 8, 12, 16]
    best_workers = run_io_experiment(args, workers_to_test)
    args.num_workers = best_workers  # fix the best worker count

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # List of optimizers to compare
    optimizers_to_test = [
        "sgd",
        "sgd_nesterov",
        "adagrad",
        "adadelta",
        "adam"
    ]

    results = []  # will store tuples of (optimizer_name, avg_train_time, final_loss, final_acc)

    for opt_name in optimizers_to_test:
        print(f"\n[C6] Running 5 epochs with optimizer = {opt_name}, num_workers = {best_workers}")
        # Rebuild model/loader each time to ensure fresh training from scratch
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=best_workers)

        model = ResNet18(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(opt_name, model.parameters())

        total_train_time = 0.0
        final_loss = 0.0
        final_acc = 0.0

        # Train for 5 epochs
        for epoch in range(5):
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            total_samples = 0

            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_start = time.perf_counter()

            # single epoch
            for batch_idx, (inputs, targets) in enumerate(train_loader):
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

            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_end = time.perf_counter()

            epoch_time = epoch_end - epoch_start
            total_train_time += epoch_time

            # final metrics for this epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_epoch_acc = epoch_correct / total_samples
            print(f"Epoch [{epoch+1}/5] - {opt_name} => "
                  f"Loss: {avg_epoch_loss:.4f}, Acc: {avg_epoch_acc:.4f}, Time: {epoch_time:.4f}s")

            # store final epoch metrics
            if epoch == 4:  # 5th epoch (index=4)
                final_loss = avg_epoch_loss
                final_acc = avg_epoch_acc

        # average training time per epoch over the 5 epochs
        avg_train_time = total_train_time / 5.0
        print(f"[C6] {opt_name}: Final Loss: {final_loss:.4f}, Final Acc: {final_acc:.4f}, "
              f"Avg Train Time (per epoch): {avg_train_time:.4f}s")

        results.append((opt_name, avg_train_time, final_loss, final_acc))

    # Print summary and short discussion
    print("\n[C6] Summary of results (optimizer, avg_epoch_time, final_loss, final_acc):")
    for r in results:
        print(f" - {r[0]}: Time={r[1]:.4f}s, Loss={r[2]:.4f}, Acc={r[3]:.4f}")

# ----------------------------------------------------------------------
# C7 experiment: BN vs. No BN with default SGD
# ----------------------------------------------------------------------
def run_c7_experiment(args):
    """
    1. Finds best num_workers from C3 (or uses known best).
    2. Uses default SGD (lr=0.1, wd=5e-4, momentum=0.9).
    3. Trains ResNet18 (with BN) vs. ResNet18NoBN (no BN) for 5 epochs each.
    4. Reports average training time per epoch, final training loss, and final accuracy.
    5. Compares speed and convergence effects with/without BatchNorm.
    """
    # 1. Determine best num_workers from C3
    workers_to_test = [0, 4, 8, 12, 16]
    best_workers = run_io_experiment(args, workers_to_test)
    args.num_workers = best_workers

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Common transform and dataset
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=best_workers)

    # Helper function to train 5 epochs with default SGD
    def train_5_epochs(model_class):
        model = model_class(num_classes=10).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        total_time = 0.0
        final_loss = 0.0
        final_acc = 0.0

        for epoch in range(5):
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            total_samples = 0

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_t = time.perf_counter()

            for batch_idx, (inputs, targets) in enumerate(train_loader):
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

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_t = time.perf_counter()

            epoch_time = end_t - start_t
            total_time += epoch_time

            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_epoch_acc = epoch_correct / total_samples

            print(f"Epoch [{epoch+1}/5] => Loss: {avg_epoch_loss:.4f}, Acc: {avg_epoch_acc:.4f}, Time: {epoch_time:.4f}s")

            # capture final epoch metrics
            if epoch == 4:  # 5th epoch
                final_loss = avg_epoch_loss
                final_acc = avg_epoch_acc

        avg_time_per_epoch = total_time / 5.0
        return avg_time_per_epoch, final_loss, final_acc

    # 2. Train WITH BatchNorm
    print("\n[C7] Training ResNet18 WITH BatchNorm...")
    bn_time, bn_loss, bn_acc = train_5_epochs(ResNet18)
    print(f"[C7] WITH BN => Avg Epoch Time: {bn_time:.4f}s, Final Loss: {bn_loss:.4f}, Final Acc: {bn_acc:.4f}")

    # 3. Train WITHOUT BatchNorm
    print("\n[C7] Training ResNet18 WITHOUT BatchNorm...")
    no_bn_time, no_bn_loss, no_bn_acc = train_5_epochs(ResNet18NoBN)
    print(f"[C7] NO BN => Avg Epoch Time: {no_bn_time:.4f}s, Final Loss: {no_bn_loss:.4f}, Final Acc: {no_bn_acc:.4f}")
    
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10 with C3, C4, C6, and C7 experiments")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to CIFAR-10 dataset")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer to use: sgd, sgd_nesterov, adam, adagrad, adadelta")
    parser.add_argument("--experiment3", action="store_true", help="Run I/O optimization experiment for C3")
    parser.add_argument("--experiment4", action="store_true", help="Run profiling experiment for C4")
    parser.add_argument("--experiment6", action="store_true", help="Run experiments with multiple optimizers (C6)")
    # New argument for C7
    parser.add_argument("--experiment7", action="store_true", help="Run BN vs. No-BN experiment (C7)")
    args = parser.parse_args()

    if args.experiment7:
        run_c7_experiment(args)
    elif args.experiment6:
        run_c6_experiment(args)
    elif args.experiment4:
        run_c4_experiment(args)
    elif args.experiment3:
        workers_to_test = [0, 4, 8, 12, 16]
        run_io_experiment(args, workers_to_test)
    else:
        train(args)
