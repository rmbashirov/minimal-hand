FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    build-essential \
    rsync \
    curl \
    wget \
    htop \
    git \
    openssh-server \
    nano \
    cmake

RUN apt-get install -y python-opencv
RUN pip install \
    pygame==1.9.4 \
    open3d==0.8.0.0 \
    transforms3d==0.3.1 \
    keyboard==0.13.4 \
    opencv_python==3.4.3.18 \
    numpy==1.18.1

RUN pip install \
    chumpy \
    scikit-image

COPY extern/azure-kinect-samples/azure_people_unpacked /opt/azure_people_unpacked
RUN cd /opt/azure_people_unpacked && ./setup.sh

# RUN apt-get install -y libglfw3-dev

USER $USERNAME:$USERNAME
# ENTRYPOINT ["fixuid", "-q"]
# CMD ["fixuid", "-q", "bash"]
WORKDIR /src
