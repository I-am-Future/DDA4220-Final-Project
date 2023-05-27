#!/bin/bash

# Convert the VOC 2012 dataset to the COCO format.

# type `conda activate xxx` (with pycocotools package) first.
# run this script under `Data-Preparation`. 
# Replace the --root_dir with your VOC2012 dataset root.

python3 voc2coco.py --root_dir ../VOCdevkit/VOC2012

