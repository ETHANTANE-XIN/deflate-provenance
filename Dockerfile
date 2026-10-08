# DeflateProvenance reference environment (proposal section III.B: "each tool's
# exact version recorded, and pinned in a container image where practical, so
# the corpus can be regenerated").
#
# This image is the environment the published corpus, model and evaluation in
# results/ were produced in; results/README.md gives the image digest and the
# exact commands.  Ubuntu packages are frozen to a snapshot date so apt-get
# resolves the same builds every time; Go, Node.js and the Rust toolchain are
# pinned by version and the downloads are checked against their published
# SHA-256 sums.
#
#   docker build -t dfp .
#   docker run --rm dfp encoders                 # list the encoder panel and versions
#   docker run --rm -v "$PWD/out:/out" dfp analyse /out/suspicious.docx -o /out
#
# Full evaluation (the command used for results/, see results/README.md):
#
#   docker run --rm -v "$PWD/out:/out" -v "$PWD/word:/word:ro" --entrypoint sh dfp \
#       /opt/dfp/scripts/reproduce.sh /out /word

FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

ARG UBUNTU_SNAPSHOT=20260927T000000Z
ARG GO_VERSION=1.24.7
ARG GO_SHA256=da18191ddb7db8a9339816f3e2b54bdded8047cdc2a5d67059478f8d1595c43f
ARG NODE_VERSION=22.22.2
ARG NODE_SHA256=88fd1ce767091fd8d4a99fdb2356e98c819f93f3b1f8663853a2dee9b438068a
ARG RUST_VERSION=1.98.0
ENV DEBIAN_FRONTEND=noninteractive \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_NOLOGO=1 \
    DFP_CACHE=/opt/dfp-cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/usr/local/go/bin:/opt/node/bin:/root/.cargo/bin:$PATH

# System encoders: 7-Zip 23.01, libarchive 3.7.2 (bsdtar), OpenJDK 21.0.12,
# .NET SDK 8.0.131, and LibreOffice 24.2.7 for real-file samples.
# Python is Ubuntu 24.04's python3 (3.12).  The snapshot service is HTTPS-only,
# so CA certificates are installed from the regular archive first.  The image
# keeps package documentation (Ubuntu's minimal image normally drops it)
# because /usr/share/doc and similar directories are real-file sources for the
# corpus (see scripts/reproduce.sh).
RUN rm -f /etc/dpkg/dpkg.cfg.d/excludes \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN apt-get update --snapshot ${UBUNTU_SNAPSHOT} \
 && apt-get install -y --no-install-recommends --snapshot ${UBUNTU_SNAPSHOT} \
      ca-certificates curl xz-utils build-essential locales \
      python3 python3-venv python3-pip \
      7zip=23.01+dfsg-11 \
      libarchive-tools=3.7.2-2ubuntu0.8 \
      openjdk-21-jdk-headless=21.0.12.1+1-1~24.04.4 \
      dotnet-sdk-8.0=8.0.131-0ubuntu1~24.04.1 \
      libreoffice-writer=4:24.2.7-0ubuntu0.24.04.6 libreoffice-calc=4:24.2.7-0ubuntu0.24.04.6 \
 && rm -rf /var/lib/apt/lists/*

# Go compress/flate and Node.js (Chromium's zlib), checksum-verified
RUN curl -fsSL -o /tmp/go.tgz https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz \
 && echo "${GO_SHA256}  /tmp/go.tgz" | sha256sum -c - \
 && tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz \
 && curl -fsSL -o /tmp/node.txz https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz \
 && echo "${NODE_SHA256}  /tmp/node.txz" | sha256sum -c - \
 && mkdir -p /opt/node && tar -C /opt/node --strip-components=1 -xJf /tmp/node.txz && rm /tmp/node.txz

# Rust toolchain (pinned) for the preflate-rs baseline (preflate-rs 0.7.6, pinned
# in tools/preflate-estimate/Cargo.toml and Cargo.lock).  Built before the rest
# of the source is copied so code changes do not rebuild it.
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION}
WORKDIR /opt/dfp
COPY tools/preflate-estimate tools/preflate-estimate
RUN cargo build --locked --release --manifest-path tools/preflate-estimate/Cargo.toml

COPY requirements.txt .
# requirements.txt pins the encoder bindings; numpy and python-docx's own
# dependencies are pinned here to the versions the published results used
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
      numpy==2.5.3 lxml==6.1.3 typing_extensions==4.16.0
ENV PATH=/opt/venv/bin:$PATH

COPY . .
RUN python -m dfp encoders

ENTRYPOINT ["python", "-m", "dfp"]
CMD ["--help"]
