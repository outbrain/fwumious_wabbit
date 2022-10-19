FROM ubuntu:18.04
ENV IMAGENAME="fwumious-builder"
ENV DEBIAN_FRONTEND=noninteractive
ARG FW_REPO_URL
ARG FW_BRANCH
ARG RUST_VERSION='1.61.0'
ARG VW_COMPILE

RUN apt-get update &&     apt-get install gcc g++ -y &&     apt-get install libboost-dev libboost-thread-dev libboost-program-options-dev libboost-system-dev libboost-math-dev libboost-test-dev zlib1g-dev -y &&     apt-get install git python3 python3-psutil python3-matplotlib lsb-release wget software-properties-common openjdk-8-jdk curl -y

ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

# Install LLVM
WORKDIR /scripts
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 13
ENV PATH="/usr/lib/llvm-11/bin/:${PATH}"

# Install newer cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt update && apt install cmake -y

# Compile fbs
WORKDIR /
RUN git clone https://github.com/google/flatbuffers.git
WORKDIR /flatbuffers
RUN git checkout tags/v1.12.0
RUN cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
RUN make
RUN make install

# Compile vw - if arg is true
RUN if [ "$VW_COMPILE" = "true" ]; then \
     git clone https://github.com/VowpalWabbit/vowpal_wabbit.git && \
     mkdir build && \
     cd build && \
     cmake .. &&\
     make vw_cli_bin -j $(nproc) ; \
 else \
     echo "skip vw compile step "; \
fi

# Get rust ecosystem operating
WORKDIR /
RUN apt-get update
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup_install.sh &&  chmod +x rustup_install.sh && ./rustup_install.sh -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup install $RUST_VERSION
ENV PATH="/root/.cargo/bin:/vowpal_wabbit/vowpalwabbit/vowpalwabbit/cli/:${PATH}"

# Conduct benchmark against vw + produce --release bin
WORKDIR /
RUN echo 'cloning $FW_BRANCH $FW_REPO_URL'
RUN git clone --branch $FW_BRANCH $FW_REPO_URL
WORKDIR /fwumious_wabbit
RUN cargo test
RUN cargo build --release
# VW - FW benchmarking currently disabled
WORKDIR /fwumious_wabbit/benchmark
CMD ["./run_with_plots.sh"]
