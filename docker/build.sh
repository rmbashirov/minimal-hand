#!/usr/bin/env bash

source source.sh
# dn="$(basename $(pwd))"
docker build -t $NAME -f Dockerfile ..
