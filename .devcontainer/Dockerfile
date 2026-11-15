# Python 3.13: the version ComfyUI recommends (the extension supports >= 3.10).
FROM mcr.microsoft.com/devcontainers/python:3-3.13-trixie

# ComfyUI release tag to install, e.g. v0.10.0 (the minimum supported version).
# Empty means the latest stable release: vX.Y.Z tags only, so peeled refs, the
# "latest" tag and pre-releases are ignored. Docker caches this step: rebuild
# without cache to pick up a newer release.
ARG COMFYUI_VERSION=
ARG COMFYUI_REPO=https://github.com/Comfy-Org/ComfyUI.git

# /ComfyUI belongs to vscode (the dev container user): ComfyUI creates its
# user, temp and output folders at runtime.
RUN set -eu; \
    version="${COMFYUI_VERSION:-$(git ls-remote --tags --refs "$COMFYUI_REPO" 'v*' \
        | sed 's#.*refs/tags/##' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1)}"; \
    test -n "$version"; \
    echo "Installing ComfyUI $version"; \
    git -c advice.detachedHead=false clone --depth 1 --branch "$version" "$COMFYUI_REPO" /ComfyUI; \
    chown -R vscode:vscode /ComfyUI

WORKDIR /ComfyUI

# CPU-only PyTorch first, so ComfyUI's requirements find torch already installed
# instead of pulling the CUDA wheels (several GB) from PyPI. The CPU index stays
# available to the second step in case a dependency forces a torch reinstall,
# and the build fails if a CUDA torch got installed anyway.
# Older releases (e.g. v0.10.0) need torchaudio, and import requests without
# listing it in their requirements.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url "$TORCH_INDEX_URL" \
    && pip install --no-cache-dir -r requirements.txt requests --extra-index-url "$TORCH_INDEX_URL" \
    && python -c "import torch; assert torch.version.cuda is None, 'CUDA torch: ' + torch.__version__"

# Debug configuration to start ComfyUI from VS Code (Run and Debug, F5)
COPY --chown=vscode:vscode launch.json /ComfyUI/.vscode/launch.json
