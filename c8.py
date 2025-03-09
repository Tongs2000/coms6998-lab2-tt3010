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

# ------------------------------------------------------------
# BasicBlock WITH BatchNorm (existing)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# ResNet18 Model (with BatchNorm)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Accuracy helper
# ------------------------------------------------------------
def compute_accuracy(outputs, targets):
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    return correct / targets.size(0)

# ------------------------------------------------------------
# Create an optimizer with consistent hyperparams
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# (C3) I/O experiment to find best num_workers
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Eager training (no compile) - with progress prints
# ------------------------------------------------------------
def train_10_epochs_eager(args, model, train_loader, device):
    """
    Trains a model for 10 epochs (GPU sync included).
    Returns:
      - time for first epoch
      - average time for epochs 6~10
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args.optimizer, model.parameters())

    epoch_times = []
    for epoch in range(10):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_t = time.perf_counter()

        batch_count = len(train_loader)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  [Eager][Epoch {epoch+1}] Processed batch {batch_idx+1}/{batch_count}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_t = time.perf_counter()
        epoch_times.append(end_t - start_t)

        print(f"Finished epoch {epoch+1} (Eager) in {epoch_times[-1]:.4f}s")

    time_first_epoch = epoch_times[0]
    time_6_to_10 = sum(epoch_times[5:10]) / 5.0  # average of epochs 6~10
    return time_first_epoch, time_6_to_10

# ------------------------------------------------------------
# Compiled training - different compile modes, with progress prints
# ------------------------------------------------------------
def train_10_epochs_compiled(args, model, train_loader, device, compile_mode):
    """
    Same as train_10_epochs_eager, but wraps the model in torch.compile
    with backend='inductor' and different 'mode' settings:
      - compile_mode = None => default
      - compile_mode = "reduce-overhead"
      - compile_mode = "max-autotune"
    """
    # Wrap model with torch.compile
    model = torch.compile(model, backend="inductor", mode=compile_mode)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args.optimizer, model.parameters())

    epoch_times = []
    for epoch in range(10):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_t = time.perf_counter()

        batch_count = len(train_loader)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  [Compiled:{compile_mode}][Epoch {epoch+1}] Processed batch {batch_idx+1}/{batch_count}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_t = time.perf_counter()
        epoch_times.append(end_t - start_t)

        print(f"Finished epoch {epoch+1} (Compiled:{compile_mode}) in {epoch_times[-1]:.4f}s")

    time_first_epoch = epoch_times[0]
    time_6_to_10 = sum(epoch_times[5:10]) / 5.0
    return time_first_epoch, time_6_to_10

# ------------------------------------------------------------
# C8 experiment: Compare Eager vs. Torch.Compile modes
# ------------------------------------------------------------
def run_c8_experiment(args):
    """
    1. Finds best num_workers from C3.
    2. Uses GPU if available.
    3. For each of the following, trains for 10 epochs on ResNet18:
       - Eager mode (no compile)
       - torch.compile(model, backend='inductor', mode=None) [Default]
       - torch.compile(..., mode='reduce-overhead')
       - torch.compile(..., mode='max-autotune')
       and measures:
         (a) Time for first epoch
         (b) Average time for epochs 6~10
    4. Prints a table summarizing the results.
    """
    # Step 1: best num_workers from C3
    workers_to_test = [0, 4, 8, 12, 16]
    best_workers = run_io_experiment(args, workers_to_test)
    args.num_workers = best_workers

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Build data loader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=best_workers)

    # We'll compare 4 settings
    modes = [
        ("Eager Mode", None, False),  # No compile
        ("Default", None, True),      # torch.compile(..., mode=None)
        ("reduce-overhead", "reduce-overhead", True),
        ("max-autotune", "max-autotune", True)
    ]

    results = []
    for mode_name, compile_mode, do_compile in modes:
        print(f"\n[C8] Running 10 epochs with mode: {mode_name}")
        model = ResNet18(num_classes=10).to(device)

        if not do_compile:
            # Eager training
            first_epoch_time, avg_6_10 = train_10_epochs_eager(args, model, train_loader, device)
        else:
            # Compiled training
            first_epoch_time, avg_6_10 = train_10_epochs_compiled(args, model, train_loader, device, compile_mode)

        results.append((mode_name, first_epoch_time, avg_6_10))

    # Print table
    print("\n=== C8 Results ===")
    print("Non Compile    Torch.Compile")
    print("--------------------------------------------------------")
    print("               Eager Mode | Default | reduce-overhead | max-autotune")
    print("Time for first epoch:")
    row_first = "  "
    row_avg   = "  "
    mode_dict = {r[0]: (r[1], r[2]) for r in results}

    # Eager => Default => reduce-overhead => max-autotune
    row_first += f"Eager={mode_dict['Eager Mode'][0]:.4f}s | "
    row_first += f"Default={mode_dict['Default'][0]:.4f}s | "
    row_first += f"reduce-overhead={mode_dict['reduce-overhead'][0]:.4f}s | "
    row_first += f"max-autotune={mode_dict['max-autotune'][0]:.4f}s"

    print(row_first)
    print("\nAverage time for epochs (6~10):")
    row_avg += f"Eager={mode_dict['Eager Mode'][1]:.4f}s | "
    row_avg += f"Default={mode_dict['Default'][1]:.4f}s | "
    row_avg += f"reduce-overhead={mode_dict['reduce-overhead'][1]:.4f}s | "
    row_avg += f"max-autotune={mode_dict['max-autotune'][1]:.4f}s"
    print(row_avg)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10 with Torch.Compile (C8)")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to CIFAR-10 dataset")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer to use: sgd, sgd_nesterov, adam, adagrad, adadelta")
    parser.add_argument("--experiment3", action="store_true", help="Run I/O optimization experiment for C3")
    parser.add_argument("--experiment4", action="store_true", help="Run profiling experiment for C4")
    parser.add_argument("--experiment6", action="store_true", help="Run experiments with multiple optimizers (C6)")
    parser.add_argument("--experiment7", action="store_true", help="Run BN vs. No-BN experiment (C7)")
    parser.add_argument("--experiment8", action="store_true", help="Run torch.compile experiment (C8)")
    args = parser.parse_args()

    if args.experiment8:
        run_c8_experiment(args)
    elif args.experiment7:
        run_c7_experiment(args)
    elif args.experiment6:
        run_c6_experiment(args)
    elif args.experiment4:
        run_c4_experiment(args)
    elif args.experiment3:
        workers_to_test = [0, 4, 8, 12, 16]
        run_io_experiment(args, workers_to_test)
