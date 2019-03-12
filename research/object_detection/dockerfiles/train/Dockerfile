FROM tensorflow/tensorflow:1.9.0-gpu-py3

RUN apt-get update
RUN apt-get -y install git-core
RUN apt-get -y install wget

RUN mkdir /tf

RUN git clone --depth 1 https://github.com/RangK/models.git && \
    mv models /tf/models

COPY pipeline.config /tf/models/research/object_detection/pipelines/faster_rcnn_nas.config
RUN mkdir /tf/models/research/object_detection/train_faster_rcnn_nas

ADD http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz /tf/models/research/object_detection/
RUN tar -xzf /tf/models/research/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz -C /tf/models/research/object_detection/
RUN rm /tf/models/research/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz

RUN cd /tf/models/research/ && \
    wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip && \
    unzip protobuf.zip

RUN apt-get -y install python3-tk
RUN apt-get -y install python3-pil
RUN apt-get -y install python3-lxml
RUN apt-get -y install python3-setuptools
RUN apt-get install python3-dev
RUN pip install cython

RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git

RUN cd cocoapi/PythonAPI && make && \
    cp -r pycocotools /tf/models/research/ && \
    cd ../../ && \
    rm -rf cocoapi

RUN curl -OL "https://github.com/google/protobuf/releases/download/v3.7.0/protoc-3.7.0-linux-x86_64.zip" && \
    unzip protoc-3.7.0-linux-x86_64.zip -d proto3 && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.7.0-linux-x86_64.zip

RUN cd /tf/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH $PYTHONPATH:/tf/models/research:/tf/models/research/slim

# Install wget (to make life easier below) and editors (to allow people to edit
# the files inside the container)
RUN apt-get install -y wget vim

WORKDIR /tf/models/research/object_detection



