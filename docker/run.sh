#!/bin/bash
docker run -it \
    --shm-size 16G \
    --log-driver=none \
    --gpus all \
    --volume=${PWD}/../data/Records:/workspace/data \
    --volume=${PWD}/..:/workspace/code \
    --entrypoint /bin/bash \
   ${USER}/clothes:latest
