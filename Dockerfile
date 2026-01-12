FROM ubuntu:22.04

# Set non-interactive frontend and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Use Alpine for smaller image (more efficient)
FROM ubuntu:22.04

# Install minimal dependencies first
RUN apt-get update && apt-get install -y \
    git build-essential \
    autoconf automake autotools-dev \
    libfl2 libfl-dev \
    bison flex gperf \
    python3 python3-pip \
    libz-dev cmake \
    gtkwave iverilog \
    help2man \
    texinfo \
    libssl-dev \
    g++ \
    make \
    perl \
    && rm -rf /var/lib/apt/lists/*

# Install Verilator with single-threaded build (reduces memory)
RUN git clone https://github.com/verilator/verilator /tmp/verilator \
    && cd /tmp/verilator \
    && git checkout v4.214 \
    && autoconf \
    && ./configure \
    # Single thread to reduce memory usage
    && make -j1 \
    && make install \
    && cd / \
    && rm -rf /tmp/verilator

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
