FROM ubuntu:22.04

# Set non-interactive frontend and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install ALL dependencies in one layer (reduces size)
RUN apt-get update && apt-get install -y \
    git build-essential \
    autoconf automake autotools-dev \
    libfl2 libfl-dev \
    bison flex gperf \
    python3 python3-pip python3-dev \
    libgoogle-perftools-dev \
    libboost-filesystem-dev libboost-iostreams-dev \
    libboost-program-options-dev libboost-system-dev libboost-thread-dev \
    libz-dev cmake \
    gtkwave iverilog \
    libncurses-dev libreadline-dev \
    perl help2man \
    g++-11 gcc-11 \
    libssl-dev zlib1g-dev \
    tcl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up gcc-11 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

# Install Verilator with FIXED version (use stable branch instead)
RUN cd /tmp \
    && git clone https://github.com/verilator/verilator \
    && cd verilator \
    # Use stable branch instead of specific tag
    && git checkout stable \
    && autoconf \
    # Configure with minimal options
    && ./configure --prefix=/usr/local \
    # Build with reduced optimization for compatibility
    && make -j$(nproc) \
    && make install \
    && cd / \
    && rm -rf /tmp/verilator

# Alternative: Install from apt (simpler but older)
# RUN apt-get install -y verilator

# Verify installation
RUN echo "=== Verilator Test ===" \
    && verilator --version \
    && echo "=== Basic Test ===" \
    && echo 'module test; initial $display("OK"); endmodule' > /tmp/test.sv \
    && verilator --cc /tmp/test.sv 2>&1 | head -5 \
    && echo "=== Installation Successful ==="

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
RUN useradd -m -s /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python3", "app.py"]
