FROM mcr.microsoft.com/devcontainers/python:1-3.12
WORKDIR /
RUN git clone --single-branch --branch $(git ls-remote --tags --sort="v:refname" https://github.com/comfyanonymous/ComfyUI.git | tail -n1 | sed 's/.*\///; s/\^{}//') https://github.com/comfyanonymous/ComfyUI
WORKDIR /ComfyUI
#RUN pip install --upgrade pip && pip install -r requirements.txt