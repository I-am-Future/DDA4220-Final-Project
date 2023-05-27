#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

set -x

# for eval, please directly use the .sh in configs
RUN_COMMAND="./configs/r50_deformable_detr.sh"


log_time=`date +"%Y-%m-%d_%T"`

log_fname='./exps/r50_deformable_detr/'$log_time'.txt'

GPUS_PER_NODE=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

python ./tools/launch.py \
    --master_port 29600 \
    --nproc_per_node ${GPUS_PER_NODE} \
    ${RUN_COMMAND} \
    > $log_fname
