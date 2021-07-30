#FROM tensorflow/tensorflow:latest-gpu-py3
FROM tensorflow/tensorflow:1.14.0-gpu-py3
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR "/root"

# install protobuf 3
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip
RUN unzip protoc-3.2.0-linux-x86_64.zip -d protoc3
RUN mv protoc3/bin/* /usr/local/bin/
RUN mv protoc3/include/* /usr/local/include/

# install pycocotool for evaluation
RUN apt-get update
RUN apt install -y git python3-tk libsm6 libxext6
RUN apt install -y jupyter
RUN apt install -y vim
RUN pip3 install cython pillow
RUN pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y python3-opencv
RUN apt install -y libgl1-mesa-glx
RUN pip3 install "opencv-python-headless==4.2.0.34"
RUN pip3 install matplotlib
RUN pip3 install protobuf
RUN pip3 install ipython
RUN pip2 install pika
