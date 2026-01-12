FROM ubuntu:22.04

# Set non-interactive frontend and timezone to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Pre-configure tzdata to avoid interactive prompts
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Now install packages
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
    help2man \
    texinfo \
    wget curl \
    libssl-dev \
    g++ \
    make \
    perl \
    tzdata \
    vim nano \
    && rm -rf /var/lib/apt/lists/*

# Configure tzdata non-interactively
RUN DEBIAN_FRONTEND=noninteractive dpkg-reconfigure tzdata

# Install Verilator with single-threaded build (reduces memory)
RUN git clone https://github.com/verilator/verilator /tmp/verilator \
    && cd /tmp/verilator \
    && git checkout v4.214 \
    && autoconf \
    && ./configure \
    && make -j1 \
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
