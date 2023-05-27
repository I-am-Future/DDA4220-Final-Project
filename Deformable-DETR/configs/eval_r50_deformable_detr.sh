#!/usr/bin/env bash
# modified by Lai, DDA4220 project use.

set -x

EXP_DIR=exps/eval_r50_deformable_detr

log_time=`date +"%Y-%m-%d_%T"`

log_fname='exps/eval_r50_deformable_detr/test_'$log_time'.txt'

export CUDA_VISIBLE_DEVICES=5

# For NORMAL evaluation (Remember to change the checkpoint!!)
python -u main.py \
    --output_dir ${EXP_DIR} \
    --resume exps/r50_deformable_detr/ckpt_sqrddetr_nq75_s13_e60.pth \
    --coco_path ./data/COCO/test \
    --eval \
    > $log_fname

