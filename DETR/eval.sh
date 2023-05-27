# evaluation
# run this at directory `DETR`!

export CUDA_VISIBLE_DEVICES=4,5,6,7


python main.py \
    --batch_size 2 \
    --no_aux_loss \
    --eval \
    --resume work_dir/ckpt_detr_nq25_e300.pth \
    --coco_path ./COCO/ \
    --output_dir ./work_dir_eval/

