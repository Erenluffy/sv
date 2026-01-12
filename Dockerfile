# Dockerfile - Production-ready with Verilator
FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git build-essential \
    autoconf automake autotools-dev \
    libfl2 libfl-dev \
    bison flex gperf \
    python3 python3-dev \
    libgoogle-perftools-dev \
    libboost-all-dev \
    libz-dev cmake \
    && rm -rf /var/lib/apt/lists/*

# Build and install Verilator
ARG VERILATOR_VERSION=v5.018
WORKDIR /build

RUN git clone https://github.com/verilator/verilator && \
    cd verilator && \
    git checkout ${VERILATOR_VERSION} && \
    autoconf && \
    ./configure --prefix=/opt/verilator \
                --enable-longtests \
                --enable-threads \
                --enable-vpi \
                --enable-coverage \
                CXXFLAGS="-O3 -march=native -DNDEBUG" && \
    make -j$(nproc) && \
    make install

# Runtime image
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libfl2 libgoogle-perftools4 \
    libboost-filesystem1.74.0 \
    libboost-iostreams1.74.0 \
    libboost-program-options1.74.0 \
    libboost-system1.74.0 \
    libboost-thread1.74.0 \
    libz1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Verilator from builder
COPY --from=builder /opt/verilator /opt/verilator

# Set environment
ENV VERILATOR_ROOT=/opt/verilator
ENV PATH="${VERILATOR_ROOT}/bin:${PATH}"
ENV MANPATH="${VERILATOR_ROOT}/share/man:${MANPATH}"
ENV LD_LIBRARY_PATH="${VERILATOR_ROOT}/lib:${LD_LIBRARY_PATH}"

# Create non-root user
RUN useradd -m -s /bin/bash veriuser
USER veriuser
WORKDIR /home/veriuser

# Install Python dependencies
COPY --chown=veriuser:veriuser requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=veriuser:veriuser app.py .
COPY --chown=veriuser:veriuser verilator_backend.py .

EXPOSE 8000
CMD ["python3", "app.py"]
