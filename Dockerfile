FROM ubuntu:18.04
ENV IMAGENAME="fwumious-builder"
ENV DEBIAN_FRONTEND=noninteractive
ARG RUST_VERSION="1.61.0"
RUN apt-get update &&     apt-get install gcc g++ -y &&     apt-get install libboost-dev libboost-thread-dev libboost-program-options-dev libboost-system-dev libboost-math-dev libboost-test-dev zlib1g-dev -y &&     apt-get install git python3 python3-psutil python3-matplotlib lsb-release wget software-properties-common openjdk-8-jdk curl -y
RUN apt-get install -y libssl-dev

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

# Get rust ecosystem operating
WORKDIR /
RUN apt-get update

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup install $RUST_VERSION
ENV PATH="/root/.cargo/bin:/vowpal_wabbit/vowpalwabbit/vowpalwabbit/cli/:${PATH}"

# Conduct benchmark against vw + produce --release bin
WORKDIR /FW
COPY . /FW

#RUN cargo test
RUN chmod +x build.sh
RUN cargo test
RUN ./build.sh
