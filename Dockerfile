ARG CUDA_VERSION=13.0.0
ARG CUDA_PIP=cu130

# Use CUDA devel image (has nvcc + headers needed to compile causal_conv1d)
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=300

# Install Python and build tools
RUN apt-get update && apt-get install -y     python3-pip     python3-dev     git     ninja-build     && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Re-declare ARG after FROM (ARGs before FROM are only available in FROM)
ARG CUDA_PIP=cu130

WORKDIR /home/sparse_caching
COPY . .

# Install sparse prefix caching and friends
RUN pip install -e . --extra-index-url https://download.pytorch.org/whl/${CUDA_PIP}

ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
RUN pip install packaging
RUN pip install causal_conv1d --no-build-isolation --extra-index-url https://download.pytorch.org/whl/${CUDA_PIP}
RUN pip install flash-linear-attention

# Verify
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.version.cuda}')"
## GPU matmul check — requires runtime GPU access, skipped during build
# RUN python -c "import torch; torch.randn( (5,5)).cuda() @ torch.randn((5,5)).cuda()"
RUN python -c "import causal_conv1d; print(f'causal_conv1d OK')"
RUN python -c "import transformers; print(f'transformers {transformers.__version__}')"
