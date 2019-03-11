FROM tensorflow/tensorflow:1.9.0-gpu-py3

RUN git clone --depth 1 https://github.com/RangK/models.git && \
    mv models /tensorflow/models

RUN apt-get install -y protobuf-compiler python-pil python-lxml python-tk



RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    make -j8 && \
    cp -r pycocotools /tensorflow/models/research && \
    cd ../../ && \
    rm -rf cocoapi

RUN curl -OL "https://github.com/google/protobuf/releases/download/v3.7.0/protoc-3.7.0-linux-x86_64.zip" && \
    unzip protoc-3.7.0-linux-x86_64.zip -d proto3 && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.7.0-linux-x86_64.zip

RUN cd /tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim

# Install wget (to make life easier below) and editors (to allow people to edit
# the files inside the container)
RUN apt-get install -y wget vim

WORKDIR /tensorflow



