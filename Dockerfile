FROM ubuntu:18.04
ENV IMAGENAME="fwumious-builder"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install gcc g++ -y && \
    apt-get install libboost-dev libboost-thread-dev libboost-program-options-dev libboost-system-dev libboost-math-dev libboost-test-dev zlib1g-dev cmake -y && \
    apt-get install git python3 python3-psutil python3-matplotlib -y

RUN apt-get install lsb-release wget software-properties-common -y

RUN apt-get install openjdk-8-jdk -y
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

# Install LLVM
WORKDIR /scripts
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 13
ENV PATH="/usr/lib/llvm-11/bin/:${PATH}"

# Compile fbs
WORKDIR /
RUN git clone https://github.com/google/flatbuffers.git
WORKDIR /flatbuffers
RUN git checkout tags/v1.12.0
RUN cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
RUN make
RUN make install

WORKDIR /
RUN git clone https://github.com/andraztori/vowpal_wabbit.git
WORKDIR vowpal_wabbit
RUN make && make install

# Compile vw - needed for benchmark
#WORKDIR /vowpal_wabbit
#RUN git clone https://github.com/VowpalWabbit/vowpal_wabbit.git
#WORKDIR /vowpal_wabbit/vowpal_wabbit
#RUN git checkout tags/8.9.2
#RUN make && make install

# Get rust ecosystem operating
RUN apt-get update
RUN apt-get -y install curl
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup_install.sh
RUN chmod +x rustup_install.sh
RUN ./rustup_install.sh -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Conduct benchmark against vw + produce --release bin
WORKDIR /fwumious_wabbit
COPY . .
RUN cargo test
RUN cd /fwumious_wabbit/benchmark && ./run_with_plots.sh
