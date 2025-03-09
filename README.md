**README.md**

This repository contains a ResNet-18 training script on CIFAR-10 with multiple experiments (C1/C2 through C8). Below are instructions on how to run each experiment using the provided command-line interface.

---

### 1. Run All Experiments in One Command

To run **all** experiments (C1/C2 through C8) in the required order with a **single** command, execute:

```bash
python3 lab2.py --all
```

This does the following:

1. **C1/C2** (Training on CPU)
2. **C3** (I/O optimization, CPU)
3. **C4** (Profiling experiment on CPU)
4. **C5** (CPU vs GPU time, reusing C4 code for GPU)
5. **C6** (Multiple optimizers, GPU)
6. **C7** (BatchNorm vs. no BatchNorm, GPU)
7. **C8** (`torch.compile` experiments, GPU)

**Note**: The script automatically forces CPU for C1–C4 and GPU for C5–C8 when using `--all`. If you have a GPU available, no additional flags are required for that overall flow.

---

### 2. Individual Experiments

- **C1/C2**: The default action (training) if no other flags are provided:
  ```bash
  python lab2.py
  ```
  This runs the ResNet-18 training on CPU.

- **C3**: I/O optimization experiment:
  ```bash
  python lab2.py --experiment3
  ```
  This measures data-loading time with varying `num_workers`. You can also add `--cuda` if you wish to run on GPU.

- **C4**: Profiling experiment:
  ```bash
  python lab2.py --experiment4
  ```
  This measures data-loading time, training time, and total time per epoch. You may include `--cuda` if you want GPU usage.

- **C5**: In this code, C5 is implemented by reusing the C4 function on GPU. If you want to run it alone:
  ```bash
  python lab2.py --experiment4 --cuda
  ```
  This yields GPU profiling times that serve as C5 results.

- **C6**: Multiple optimizers (SGD, Adam, etc.):
  ```bash
  python lab2.py --experiment6 --cuda
  ```
  This compares training times, losses, and accuracies for each optimizer.

- **C7**: BatchNorm vs. no BatchNorm:
  ```bash
  python lab2.py --experiment7 --cuda
  ```
  This trains ResNet-18 with BN vs. ResNet-18 without BN for 5 epochs each, comparing speed and convergence.

- **C8**: `torch.compile` with Inductor backend:
  ```bash
  python lab2.py --experiment8 --cuda
  ```
  This measures the time for the first epoch and the average time for epochs 6–10 under different compile modes.

---

### 3. Additional Arguments

- `--cuda`: Forces the script to use GPU if available.  
- `--data_path`: Path to the CIFAR-10 dataset (default is `./data`).  
- `--num_workers`: Number of data loader workers (default is 2).  
- `--optimizer`: Which optimizer to use (e.g., `sgd`, `adam`).

**Example**:
```bash
python lab2.py --experiment6 --cuda --data_path /path/to/cifar --optimizer adam
```

---

### 4. Notes

1. **C5** is reusing the same logic as C4, but on GPU, so when running `--all`, the code forcibly switches to GPU for that step.
2. If you only want to see the C5 GPU result alone, you can run `python lab2.py --experiment4 --cuda`.
3. The code in `--all` mode automatically handles CPU vs GPU switching for each stage, so no extra flags are needed to fulfill the assignment requirements for each experiment.