## Training Environment

| Component | Specification |
|---|---|
| OS | Ubuntu 24.04 |
| CPU | Intel Core i9-10980XE |
| GPU | NVIDIA GeForce RTX 3090 × 2 |
| RAM | 128 GB |
| NVIDIA Driver | 595-open |
| Python | 3.11 |
| PyTorch | 2.11.0 |
| Torchvision | 0.26.0 |
| CUDA Runtime | 12.8 |

## Environment Setup

### 1. Create a Conda environment

```bash
conda create -n semantic_line python=3.11
conda activate semantic_line
```

### 2. Install PyTorch

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install dependencies

```bash
python -m pip install -r requirements-server.txt
```

### 4. Verify CUDA availability

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## Environment Files

- `environment.yml`: Conda environment specification.
- `requirements-server.txt`: Python package versions used on the training server.

The dataset and model checkpoints are managed separately from this repository.