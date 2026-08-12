# Agentic RL Research Environment
# CUDA 12.1.1 + Python 3.10 + PyTorch 2.4 + vLLM + veRL + tau-bench + SwanLab
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------------ #
#  System packages
# ------------------------------------------------------------------ #
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils \
    python3-pip git curl wget vim build-essential \
    openssh-client rsync \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# ------------------------------------------------------------------ #
#  Python packages — core ML stack
# ------------------------------------------------------------------ #
RUN pip install --upgrade pip==24.2 setuptools==70.0.0 wheel==0.43.0

# PyTorch 2.4.0 + CUDA 12.1
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Flash Attention 2
RUN pip install flash-attn==2.6.3 --no-build-isolation

# vLLM (compatible with PyTorch 2.4 + CUDA 12.1)
RUN pip install vllm==0.5.5

# ------------------------------------------------------------------ #
#  Python packages — research tools
# ------------------------------------------------------------------ #
# tau-bench
RUN pip install tau-bench

# SwanLab for experiment tracking
RUN pip install swanlab

# Hugging Face Hub for model downloads
RUN pip install huggingface_hub==0.24.6

# Transformers + tokenizers
RUN pip install transformers==4.44.2 tokenizers==0.19.1

# Utilities
RUN pip install pyyaml==6.0.2 numpy==1.26.4 scipy==1.14.1

# ------------------------------------------------------------------ #
#  veRL (from source, pinned commit)
# ------------------------------------------------------------------ #
RUN git clone https://github.com/volcengine/verl.git /opt/verl && \
    cd /opt/verl && \
    git checkout v0.3.0 && \
    pip install -e .

# ------------------------------------------------------------------ #
#  Workspace
# ------------------------------------------------------------------ #
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install project as package
RUN pip install -e .

# Environment variables
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV TOKENIZERS_PARALLELISM=false

# Default command
CMD ["/bin/bash"]
