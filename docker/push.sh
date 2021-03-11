#!/usr/bin/env bash

source source.sh

docker tag $NAME $HEAD_NAME
docker push $HEAD_NAME

