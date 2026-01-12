FROM ubuntu:22.04

# Set non-interactive frontend and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install ALL required dependencies
RUN apt-get update && apt-get install -y \
    git build-essential \
    autoconf automake autotools-dev \
    libfl2 libfl-dev \
    bison flex gperf \
    python3 python3-pip \
    libgoogle-perftools-dev \
    libboost-all-dev \
    libz-dev cmake \
    gtkwave iverilog \
    # Essential for Verilator build
    help2man \
    texinfo \
    wget curl \
    libssl-dev \
    g++ \
    make \
    perl \
    && rm -rf /var/lib/apt/lists/*

# Install Verilator v5.008 (with all dependencies now)
RUN git clone https://github.com/verilator/verilator /tmp/verilator \
    && cd /tmp/verilator \
    && git checkout v5.008 \
    && autoconf \
    && ./configure \
    && make -j$(nproc) \
    && make install \
    && cd / \
    && rm -rf /tmp/verilator

# Verify installation
RUN verilator --version

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create app directories
RUN mkdir -p /app /tmp/waveforms /tmp/coverage /tmp/logs
WORKDIR /app

# Copy application files
COPY app.py .
COPY problems.json .
COPY problems_sv.json .

# Non-root user
RUN useradd -m -s /bin/bash appuser
USER appuser

EXPOSE 8000
CMD ["python3", "app.py"]
