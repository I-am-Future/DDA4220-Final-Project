# training 
# run this at directory `DETR`!

log_time=`date +"%Y-%m-%d_%T"`

log_fname='./work_dir/'$log_time'.txt'

export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --coco_path ./COCO/ \
    --output_dir ./work_dir/ \
    > $log_fname

