# DeflateProvenance reference environment (proposal section III.B: "each tool's
# exact version recorded, and pinned in a container image where practical, so
# the corpus can be regenerated").
#
# The versions below are the ones the published corpus and evaluation were
# produced with (see results/manifest.json).  Ubuntu packages are frozen to a
# snapshot date so `apt-get` resolves exactly the same builds; Go and Node.js
# come from their official release archives because Ubuntu 24.04 ships older
# ones.
#
#   docker build -t dfp .
#   docker run --rm -v "$PWD/out:/out" dfp evaluate -o /out --libreoffice 10 --python-docx 20
#
# Note: this file has not been test-built on the machine that produced the
# results (no Docker daemon was available there); the same versions were
# installed natively.

FROM ubuntu:24.04

ARG UBUNTU_SNAPSHOT=20260927T000000Z
ARG GO_VERSION=1.24.7
ARG NODE_VERSION=22.22.2
ENV DEBIAN_FRONTEND=noninteractive \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_NOLOGO=1 \
    DFP_CACHE=/opt/dfp-cache \
    PATH=/usr/local/go/bin:/opt/node/bin:/root/.cargo/bin:$PATH

# System encoders: 7-Zip 23.01, libarchive 3.7.2 (bsdtar), OpenJDK 21.0.10,
# .NET SDK 8.0.131 (runtime 8.0.31), LibreOffice 24.2.7 for real-file samples.
RUN apt-get update --snapshot ${UBUNTU_SNAPSHOT} \
 && apt-get install -y --no-install-recommends --snapshot ${UBUNTU_SNAPSHOT} \
      ca-certificates curl xz-utils build-essential \
      python3 python3-venv python3-pip \
      7zip=23.01+dfsg-11 \
      libarchive-tools=3.7.2-2ubuntu0.8 \
      openjdk-21-jdk-headless=21.0.10+7-1~24.04 \
      dotnet-sdk-8.0=8.0.131-0ubuntu1~24.04.1 \
      libreoffice-writer libreoffice-calc \
 && rm -rf /var/lib/apt/lists/*

# Go compress/flate and Node.js (Chromium's zlib 1.3.1-e00f703)
RUN curl -fsSL https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz | tar -C /usr/local -xz \
 && mkdir -p /opt/node \
 && curl -fsSL https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz \
    | tar -C /opt/node --strip-components=1 -xJ

# Rust toolchain for the preflate-rs baseline (preflate-rs 0.7.6, pinned in Cargo.toml)
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal

WORKDIR /opt/dfp
COPY requirements.txt .
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
ENV PATH=/opt/venv/bin:$PATH

COPY . .
RUN cargo build --release --manifest-path tools/preflate-estimate/Cargo.toml \
 && python -m dfp encoders

ENTRYPOINT ["python", "-m", "dfp"]
CMD ["--help"]
