#!/bin/bash

app=$(pwd)

docker run --name pmunet -it --rm \
    --net=host --ipc=host \
    --gpus "all" \
    -v ${app}:/app \
    pmunet
