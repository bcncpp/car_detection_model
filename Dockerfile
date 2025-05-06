FROM nvidia/cuda:12.6.0-base-ubuntu22.04
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
# Install Go
ENV GOLANG_VERSION=1.21.5
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
ENV PATH="${PATH}:/home/${USERNAME}/go/bin"
RUN apt-get update && apt -y install curl
RUN curl -sSL "https://go.dev/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz \
    && ln -s /usr/local/go/bin/go /usr/local/bin/go \
    && ln -s /usr/local/go/bin/gofmt /usr/local/bin/gofmt
# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Copy the application into the container.
COPY . /app
# Install the application dependencies.
# install the bombardier
RUN go install github.com/codesenberg/bombardier@latest
WORKDIR /app
EXPOSE 8080
RUN uv add torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN uv sync
CMD uv run uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4 
