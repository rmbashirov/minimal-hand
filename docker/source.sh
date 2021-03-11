#!/usr/bin/env bash

PARAMS="--net=host --ipc=host -u $(id -u ${USER}):$(id -g ${USER})"
NAME="tf-1.14.0-gpu-py3-jupyter"
HEAD_NAME="airuhead01:5000/${NAME}"
VOLUMES="-v $PWD/..:/src -v /Vol0:/Vol0 -v /Vol1:/Vol1"
