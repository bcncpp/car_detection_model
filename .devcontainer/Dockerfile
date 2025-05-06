# Base image with common tools
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Set python

RUN apt-get update && \
    apt-get install -y \
        git \
        software-properties-common \
        wget \
        curl  \
        python3-pip \
        python3-dev \
        apt-utils \
        python3-opencv \
        libglib2.0-0 \
        zlib1g \
        g++  \
        curl


# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set environment variables for Go installation
ENV GOLANG_VERSION=1.21.5

# Install Go
RUN curl -sSL "https://go.dev/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz \
    && ln -s /usr/local/go/bin/go /usr/local/bin/go \
    && ln -s /usr/local/go/bin/gofmt /usr/local/bin/gofmt

# Install kubectl (Kubernetes CLI)
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && mv kubectl /usr/bin

# Install Helm (Kubernetes package manager)
RUN wget https://get.helm.sh/helm-v3.17.3-linux-amd64.tar.gz -O /tmp/helm.tar.gz \
    && tar -xvzf /tmp/helm.tar.gz -C /tmp/ \
    && mv /tmp/linux-amd64/helm /usr/bin/helm \
    && rm -rf /tmp/helm.tar.gz /tmp/linux-amd64

# Install Python 3, pip, and related tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    && apt-get clean

# Install Node.js (for Reveal.js)
ENV NODE_VERSION=20
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm

# Create a non-root user 'vscode' with UID and GID as 1000
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
ENV PATH="${PATH}:/home/${USERNAME}/go/bin"
#RUN mkdir -p ${USERNAME} && chmod ${USER_GID} /home/${USERNAME}
RUN useradd -u ${USER_UID} ${USERNAME}
# Switch to the non-root user

# Create the Go workspace directory for the user
RUN mkdir -p "/home/${USERNAME}/go"
# Install golangci-lint (Go linting tool) for the user
RUN curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/HEAD/install.sh | sh -s -- -b $(go env GOPATH)/bin v2.0.2

# Set working directory to the user's home
WORKDIR "/home/${USERNAME}"

# Install Bombardier (Go HTTP benchmarking tool)
RUN go install github.com/codesenberg/bombardier@latest
# Clean up any unused cache and temporary files
RUN rm -rf /var/lib/apt/lists/*
# The container will run as a non-root user for security purposes
# Upgrade pip
RUN python3 -m pip install --upgrade pip
# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
# Set the default shell to bash
RUN chown -R vscode "/home/${USERNAME}"
USER $USERNAME

SHELL ["/bin/bash", "-c"]

