# README

By [Lai Wei](https://github.com/I-am-Future), [Huihan Yang](https://github.com/foxintohumanbeing), [Rongxiao Qu](). DDA4220 Final Project, Spring 2023.

## 0. File Structure

### 0.1 Project root

+ `Data-Preparation`: please refer to the Section 1.1 below.
+ `DETR`: the implementation of DETR we used from https://github.com/facebookresearch/detr.
+ `Deformable-DETR`: the implementation of Deformable-DETR we used from https://github.com/fundamentalvision/Deformable-DETR. We modified some of the files inside it for implementing strategies, result analysis...
+ `Post-Analysis`: includes the scripts we use for plotting.

### 0.2 OneDrive root

All of our experiment weights, training logging, testing logging are at [OneDrive Link](https://cuhko365-my.sharepoint.com/:f:/g/personal/120090485_link_cuhk_edu_cn/Eq_qt23vsLJAi-eJNlTUhjwBzmT139TiyoKf7vEnoCKoDg?e=3wCdPa). In details,

+ Filename begins with`ckpt_` is a weight file. Filename begins with`log_` is a training logging file. Filename begins with`test_` is a testing logging file.
+ The second part, `detr`, `ddetr`, `sqrddetr` indicate the model structure. 
+ `nq` means number of queries.
+ last `e65` means the result at epoch 65.
+ the middle optional segment, like `lincoef`, is special experiment setting (linear stage weight coefficient)
+ E.g., `ckpt_ddetr_nq75_noauxloss_e65.pth` is the checkpoint file of a Deformable DETR, with number of queries 75, no auxiliary loss during training, weights saved at epoch 65.

## 1. Setup

### 1.1 Data

For more about the dataset information, please refer to our report.

The data should be in the COCO format, so that we can benefit from the existing Dataset and evaluation tools. We provide the **code script** to convert the VOC dataset to the COCO format.

On the other way, we also provide the converted COCO format data as in the [OneDrive Link](https://cuhko365-my.sharepoint.com/:u:/g/personal/120090485_link_cuhk_edu_cn/EXF-XYXpeUhDso64knZbP2cB7bz6snA8dMPKfR16ANYe1Q?e=3neaZO). (If the link dead, please don't be hesitated to contact us)

The dataset directory should be like below. (Put the training data in `train2017`, validation data in `val2017`, and the test data in `test`. The test directory follows the similar structure so that it can be "cheat" the dataset loader and the program would operate correctly.)

```
`-- COCO/
    |-- train2017/JPEGImages        (train data image)
    |-- val2017/JPEGImages          (val data image)
    |-- annotations/
    |   |-- instances_train2017.json (train data annotaion)
    |   `-- instances_val2017.json   (val data annotation)
    `-- test
        |-- val2017/JPEGImages      (test data image)
        |-- annotations/
            |-- instances_train2017.json (FAKE use)
            `-- instances_val2017.json (test data annotaion)
```

+ Then, for DETR, put the dataset `COCO` :

```
DETR_root/
      |-- COCO/
      |-- other things...
```

+ Deformable DETR

```
DeformableDETR_root/
  |-- data/
  |   `-- COCO/
  |-- other things...
```


### 1.2 Environment

In our practice, the Deformable DETR's environment can be well used in the DETR's code. So, we only need to setup the environment for the Deformable DETR. Please follow the setup guide of the original [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). We quote them here:

> * Linux, CUDA>=9.2, GCC>=5.4
>
> * Python>=3.7
>
>     We recommend you to use Anaconda to create a conda environment:
>   ```bash
>   conda create -n deformable_detr python=3.7 pip
>   ```
>   Then, activate the environment:
>   ```bash
>   conda activate deformable_detr
>   ```
>
> * PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))
>
>     For example, if your server's CUDA toolkit (with `nvcc`) version is 9.2, you could install pytorch and torchvision as following:
>     ```bash
>     conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
>     ```
>
> * Other requirements
>     ```bash
>     pip install -r requirements.txt
>     ```
>
> Besides, we need to compile a cuda operator locally.
>
> ```bash
> # at the root directory of the Deformable-DETR, not the project root
> cd ./models/ops
> sh ./make.sh
> # unit test (should see all checking is True)
> python test.py
> ```

**Our team's extra notes:**

+ For higher version of `torchvision>=0.10`, there is a bug in recognizing the version and import the correct things. The error looks like "cannot import name `_NewEmptyTensorOp` from `torchvision.ops.misc`". Please refer to the solution [Here](https://blog.csdn.net/y96q1023/article/details/78498894). (In our code base, we **already updated** the `util/misc.py`, so it shouldn't be a problem)
+ You need to make sure the PyTorch's CUDA version matches the CUDA version that compile the operators locally. Please refer to [Pytorch C packages CUDA Installation Mismatch Problems](https://i-am-future.github.io/2023/04/27/pytorch-C-packages-installation-failed-problems/).
+ For the GPU within 24GB Memory, you may fail to run the `test.py` under `./models/ops` successfully because of CUDA out of Memory when running size 2048. Successfully running to size 1025 is fine. 



## 2. Run

### 2.1 Train-DETR

We only have **1** Experiment on DETR. To train the DETR from scratch, we need to:

```sh
conda activate xxx    # your environment name
cd DETR
# before running the `train.sh`:
# set up the CUDA_VISIBLE_DEVICES if necessary.
# set the nproc_per_node number if necessary.
sh train.sh
```

The checkpoint and the logging will be stored at `DETR/work_dir`.

### 2.2 Evaluation-DETR

To evaluate the model, you need first download the ckpt provided from [OneDrive Link](https://cuhko365-my.sharepoint.com/:f:/g/personal/120090485_link_cuhk_edu_cn/Eq_qt23vsLJAi-eJNlTUhjwBzmT139TiyoKf7vEnoCKoDg?e=3wCdPa). The `ckpt_detr_nq25_e300.pth`. Copy it under `DETR/work_dir`. Modify the path in `eval.sh`. Then run:

```sh
conda activate xxx    # your environment name
cd DETR
# before running the `eval.sh`:
# set up the CUDA_VISIBLE_DEVICES if necessary.
sh eval.sh
```

The evaluation results will be displayed in the terminal. We also provide our result from [OneDrive Link](https://cuhko365-my.sharepoint.com/:f:/g/personal/120090485_link_cuhk_edu_cn/Eq_qt23vsLJAi-eJNlTUhjwBzmT139TiyoKf7vEnoCKoDg?e=3wCdPa). 

### 2.3 Train-Deformable DETR

We have **9** experiments on DETR. To train the Deformable DETR from scratch, we need to:

First, set the desired training configurations. You can modify the `Deformable-DETR/configs/r50_deformable_detr.sh` as you want. We provide some sample training command, you can comment/uncomment some of them.

Second, type:

```sh
conda activate xxx    # your environment name
cd Deformable-DETR
# before running the `tools/run_dist_launch.sh`:
# set up the CUDA_VISIBLE_DEVICES if necessary.
# set the GPUS_PER_NODE number if necessary.
sh tools/run_dist_launch.sh
```

The checkpoint and the logging will be stored at `Deformable-DETR/exps/r50_deformable_detr`.

### 2.4 Evaluation-Deformable DETR

To evaluate the model, you need first download the ckpt provided from [OneDrive Link](https://cuhko365-my.sharepoint.com/:f:/g/personal/120090485_link_cuhk_edu_cn/Eq_qt23vsLJAi-eJNlTUhjwBzmT139TiyoKf7vEnoCKoDg?e=3wCdPa). There are 9 possible weights to use. Copy it under `Deformable-DETR/exps/r50_deformable_detr`. Modify the path in `configs/eval_r50_deformable_detr.sh`Then run:

```sh
conda activate xxx    # your environment name
cd Deformable-DETR
# before running the `configs/eval_r50_deformable_detr.sh`:
# set up the CUDA_VISIBLE_DEVICES if necessary.
sh configs/eval_r50_deformable_detr.sh
```

The evaluation results will be stored at `Deformable-DETR/exps/eval_r50_deformable_detr`. We also provide our result from [OneDrive Link](https://cuhko365-my.sharepoint.com/:f:/g/personal/120090485_link_cuhk_edu_cn/Eq_qt23vsLJAi-eJNlTUhjwBzmT139TiyoKf7vEnoCKoDg?e=3wCdPa). 



## 3. Analysis

### 3.1 Model inspector

We hack the model's criterion function to get all the immediate data to calculate the TP Fading rate, FP Exacerbation rate. The analysis code is at `Deformable-DETR/models/inspect_dec_stage.py`.

To enable this feature, go to the `Deformable-DETR/models/deformable_detr.py`, uncomment the line 367, and follow the steps in Sec. 2.4. The result will be at `Deformable-DETR/exps/evaluating_stages/ddetr`.

### 3.2 Plotter

We plot the validation accuracy figure on the report by the code provided by the `Post-Analysis` directory. Run the `Post-Analysis/plot_valacc_curves.py` file. You may need to save the logging at specific position  (`Deformable-DETR/exps/r50_deformable_detr`) first.

## 4. Trouble Shooting

If you meet any problem running the code, this may be the environment problems (missing packages) or path problems (wrong relative path). You may try to fix it first. If the problem still exists in reproducing the code, please contact us. 
