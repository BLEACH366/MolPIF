#!/usr/bin/env bash
set -e

# ========== 配置部分 ==========
ENV_NAME=MolPIF
PYTHON_VER=3.11

# ========== Step 1: 安装 mamba ==========
if ! command -v mamba &> /dev/null; then
    echo "[INFO] Installing mamba..."
    conda install -y -n base -c conda-forge mamba
fi

# ========== Step 2: 创建 conda 环境 ==========
if conda env list | grep -q "$ENV_NAME"; then
    echo "[INFO] Conda environment '$ENV_NAME' already exists."
else
    echo "[INFO] Creating environment '$ENV_NAME' with Python $PYTHON_VER..."
    mamba create -y -n $ENV_NAME python=$PYTHON_VER
fi

# ========== Step 3A: 安装 PyTorch + CUDA ==========
echo "[INFO] Installing PyTorch stack..."
mamba install -y -n $ENV_NAME -c pytorch -c nvidia \
    pytorch=2.5.0 \
    pytorch-cuda=12.4 \
    torchvision=0.20.0 \
    torchaudio=2.5.0

# ========== Step 3B: 安装分子模拟工具 ==========
echo "[INFO] Installing molecular modeling tools..."
mamba install -y -n $ENV_NAME -c conda-forge -c mx \
    openbabel=3.1.1 \
    spyrmsd=0.9.0 \
    vina=1.2.5 \
    six=1.17.0 \
    reduce=3.24

# ========== Step 3C: 安装 PyTorch Lightning ==========
echo "[INFO] Installing PyTorch Lightning..."
mamba install -y -n $ENV_NAME -c conda-forge \
    pytorch-lightning=2.5.5 

# ========== Step 4: 安装 pip 依赖 ==========
echo "[INFO] Installing pip dependencies..."
mamba run -n $ENV_NAME pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

mamba run -n $ENV_NAME pip install -U --no-cache-dir \
    absl-py==2.2.2 \
    easydict==1.13 \
    fire==0.7.0 \
    imageio==2.37.0 \
    lmdb==1.6.2 \
    matplotlib==3.10.5 \
    meeko==0.1.dev3 \
    numpy==1.26.4 \
    oddt==0.5 \
    overrides==7.7.0 \
    pdb2pqr==3.6.2 \
    rdkit==2023.9.5 \
    scikit-learn==1.3.0 \
    torch-geometric==2.6.1 \
    torchdiffeq==0.1.1 \
    wandb==0.19.10 \
    posecheck==1.3.1 

mamba run -n $ENV_NAME python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

echo "[INFO] Environment setup completed successfully!"
echo "To use it, run: conda activate $ENV_NAME"

