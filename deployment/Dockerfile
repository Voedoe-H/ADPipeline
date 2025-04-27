FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY externals/ externals/
COPY inferenceServer.cpp .
COPY pytorchmodel_v2.onnx .
COPY CMakeLists.txt .

RUN mkdir build && cd build && cmake .. && make

EXPOSE 8080

WORKDIR /app/build

COPY pytorchmodel_v2.onnx .

CMD ["./InferenceServer"]
