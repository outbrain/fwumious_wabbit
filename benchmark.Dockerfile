FROM ubuntu:18.04
ENV IMAGENAME="fwumious-builder"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install git -y

RUN apt-get update && \
    apt-get install gcc g++ -y && \
    apt-get install libboost-dev libboost-thread-dev libboost-program-options-dev libboost-system-dev libboost-math-dev libboost-test-dev zlib1g-dev -y && \
    apt-get install git python3 python3-psutil python3-matplotlib -y

RUN apt-get install cmake -y

ENV PATH="/usr/bin/cmake/bin:${PATH}"

RUN apt-get install lsb-release wget software-properties-common -y

RUN apt-get install openjdk-8-jdk -y

ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

WORKDIR /

RUN apt-add-repository ppa:hnakamur/flatbuffers
RUN apt update
RUN apt install -y flatbuffers-compiler=1.11.0+dfsg1-1.1ubuntu1ppa1~bionic

WORKDIR /scripts
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 13
ENV PATH="/usr/lib/llvm-11/bin/:${PATH}"

WORKDIR /vowpal_wabbit
RUN git clone https://github.com/VowpalWabbit/vowpal_wabbit.git
WORKDIR /vowpal_wabbit/vowpal_wabbit
RUN git checkout tags/8.9.2
RUN make && make install

RUN apt install -y musl-tools
RUN which g++
RUN ln -s /usr/bin/g++ /usr/bin/musl-g++
RUN apt-get update
RUN apt-get -y install curl
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup_install.sh
RUN chmod +x rustup_install.sh
RUN ./rustup_install.sh -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup target add x86_64-unknown-linux-musl

WORKDIR /
RUN git clone https://github.com/outbrain/fwumious_wabbit.git
WORKDIR /fwumious_wabbit
COPY . .
RUN mkdir -p java/src/com/outbrain/fw
RUN cargo test
RUN ./run_one.sh; exit 0
RUN cd /fwumious_wabbit/benchmark && ./run_with_plots.sh