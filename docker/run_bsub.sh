#!/usr/bin/env bash

bsub -q normal -gpu "num=1:mode=shared:j_exclusive=yes" -oo $PWD/log.txt -m airugpub01 \
    'export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',') && \
    source ./source.sh && \
    docker pull $HEAD_NAME && \
    nvidia-docker run -t $PARAMS $VOLUMES $HEAD_NAME sleep infinity'
