#!/usr/bin/env bash
# modified by Lai, DDA4220 project use.

set -x

EXP_DIR=exps/r50_deformable_detr
PY_ARGS=${@:1}


# Normal train, with sqr enabled
# python -u main.py \
#     --output_dir ${EXP_DIR} \
#     --cache_mode \
#     --sqr \
#     --batch_size 2 \
#     --coco_path data/COCO/ \
#     ${PY_ARGS}


# Normal train, WITHOUT sqr enabled
python -u main.py \
    --output_dir ${EXP_DIR} \
    --cache_mode \
    --batch_size 2 \
    --coco_path data/COCO/ \
    ${PY_ARGS}


# run with aux loss on, but weight distributed
# python -u main.py \
#     --output_dir ${EXP_DIR} \
#     --cache_mode \
#     --batch_size 2 \
#     --coco_path data/COCO/ \
#     --stage_weights 0.1666 0.3333 0.5000 0.6666 0.8333 \
#     ${PY_ARGS}


# Optional components: to alter the number of queries
    # --num_queries 150 \


