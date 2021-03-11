#!/usr/bin/env bash

source source.sh

# override VOLUMES variable specified in source.sh
VOLUMES="-v $PWD/..:/src -v /mnt/hdd10:/mnt/hdd10 -v /mnt/dat:/mnt/dat -v /home/renat:/home/renat"
OTHER="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"

docker run -ti $PARAMS $VOLUMES $OTHER $NAME $@
