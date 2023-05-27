import torch
from typing import List, Dict
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import torch.nn.functional as F
import pickle

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import matplotlib.cm as cm


class_dict = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
    21: "background"
}


def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def tensor_to_numpy(tensor):
    return  np.array(tensor.detach().cpu())

def coco_to_pixel_bbox(coco_bbox, width, height):
    x_center, y_center, w, h = coco_bbox
    x_center *= width
    y_center *= height
    w *= width
    h *= height
    x = x_center - w / 2
    y = y_center - h / 2
    return x, y, w, h


def plot_bounding_boxes(image_path, predictions, targets, save_dir=None, save_name=None, dpi=96):
    image = load_image(image_path)
    width, height = image.size
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    # fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    # fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(image)

    # Get a colormap with enough colors for the unique labels
    num_labels = len(set(predictions["labels"]).union(set(targets["labels"])))
    # colormap = cm.get_cmap("tab20", num_labels)

    # Draw target bounding boxes
    for bbox, label in zip(targets["boxes"], targets["labels"]):
        x, y, w, h = coco_to_pixel_bbox(bbox, width, height)
        # label_color = colormap(label)
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor="none")
        ax.add_patch(rect)
        label = f"label: {class_dict[label]}"
        # ax.text(x, y, label, color="r", backgroundcolor="white", fontsize=10, fontweight="bold")

    # Draw predicted bounding boxes
    for bbox, label in zip(predictions["boxes"], predictions["labels"]):
        x, y, w, h = coco_to_pixel_bbox(bbox, width, height)
        # label_color = colormap(label)
        # label_color_pred = tuple([0.5 * c for c in label_color])  # Make the color lighter for predictions
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor="none")
        ax.add_patch(rect)
        label = f"prediction: {class_dict[label]}"
        # ax.text(x, y, label, color="g", backgroundcolor="white", fontsize=10, fontweight="bold")

    ax.axis('off')  # Remove axis

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Construct the save path using the save_name or the original image filename
        save_name = save_name or os.path.splitext(os.path.basename(image_path))[0] + '.png'
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    plt.close(fig)


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    inter_x1 = torch.max(x1, x2)
    inter_y1 = torch.max(y1, y2)
    inter_x2 = torch.min(x1 + w1, x2 + w2)
    inter_y2 = torch.min(y1 + h1, y2 + h2)

    inter_area = torch.max(inter_x2 - inter_x1, torch.tensor(0.0)) * torch.max(inter_y2 - inter_y1, torch.tensor(0.0))
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    union_area = bbox1_area + bbox2_area - inter_area

    iou_score = inter_area / union_area
    return iou_score


def calculate_image_iou(predictions, targets):
    # Convert to tensors if they are numpy arrays
    if isinstance(predictions["boxes"], np.ndarray):
        predictions["boxes"] = torch.tensor(predictions["boxes"])
    if isinstance(targets["boxes"], np.ndarray):
        targets["boxes"] = torch.tensor(targets["boxes"])

    iou_scores = []

    for pred_box in predictions["boxes"]:
        max_iou = 0
        for tgt_box in targets["boxes"]:
            iou_score = iou(pred_box, tgt_box)
            max_iou = max(max_iou, iou_score)

        iou_scores.append(max_iou)

    # Calculate mean IoU for the image
    mean_iou = sum(iou_scores) / len(iou_scores)

    return mean_iou



def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


class MyInspector():
    def __init__(self) -> None:
        self.records = []
        self.n_tp = [0, 0]
        self.n_fp = [0, 0]
        self.n_tp_f = [0, 0]
        self.n_fp_e = [0, 0]
        self.out_dir = "exps/evaluating_stages/ddetr/"

    def dump(self):
        ''' Dump the all statistics. '''
        # print(self.records)
        dump_dir = self.out_dir + "stages_stat1.pkl"
        with open(dump_dir, 'wb') as f:
            pickle.dump(self.records, f)

    def inspect_batch(self, auxoutputs: List[Dict], finaloutputs, targets, matcher):
        ''' Inspect a batch of data, for specific use.

        Inputs:
            auxoutputs: List[Dict], len = Batchsize, which is a list of Dicts, 
                contains logits of each layer. {'pred_logits' and 'pred_boxes'}

            finaloutputs: Dict, logits of last layer. {'pred_logits' and 'pred_boxes'}

            targets: List[Dict], len = Batchsize, which is a list of Dicts, contains coco-format ground truth

            matcher: A matcher that can match (targets, groundtruth) and (outputs, prediction).
            
        Usage Guidence:
            to iterate all layer's outputs:
                `for i, output in enumerate(all_outputs):` 
                output['pred_logits'].shape  is  B x Q x C
                output['pred_boxes'].shape   is  B x Q x 4
       
        '''
        # get all outputs.
        all_outputs = auxoutputs + [finaloutputs]

        B = finaloutputs['pred_logits'].shape[0]

        # iterate all images in a batch
        for imgidx in range(B):
            print('----------IMG----------')
            # iterate all layers
            for lidx, output in enumerate(all_outputs):
                src_logits = output['pred_logits'][imgidx].unsqueeze(0)
                src_boxes = output['pred_boxes'][imgidx].unsqueeze(0)

                # `indices` is used for recording the closest query to the ground truth
                indices = matcher(output, targets)

                targetimg = [targets[imgidx]]
                indicesimg = [indices[imgidx]]

                ### investigating on class label

                # `idx` is used for indexing (getting used query)
                # looks like: (tensor([0, 0]), tensor([ 7, 22])), to get query 7, 22
                idx = _get_src_permutation_idx(indicesimg)
                # target_classes_o is used for getting accuracy statistics
                # looks like: tensor([7, 2], device='cuda:0'), query 7, 2's correct answer
                target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targetimg, indicesimg)])

                # then we can calulate the acc by query logits and target
                query_logits = src_logits[idx]
                print(query_logits.argmax(axis=1), target_classes_o)
                print(f'stage {lidx} acc:', (accuracy(src_logits[idx], target_classes_o)[0]).cpu().item()  )

                ### investigating on bbox

                ## Get the target boxes
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targetimg, indicesimg)], dim=0)
                query_boxes = src_boxes[idx]
                loss_bbox = F.l1_loss(query_boxes, target_boxes, reduction='none')
                print(f'stage {lidx} loss:', loss_bbox.sum(axis=1)  )


    def inspect_batch_query(self, auxoutputs: List[Dict], finaloutputs, targets, matcher, print_imgs=True, output_dir=None):
        ''' Inspect a batch of data, for specific use.

        Inputs:
            auxoutputs: List[Dict], len = Batchsize, which is a list of Dicts, 
                contains logits of each layer. {'pred_logits' and 'pred_boxes'}

            finaloutputs: Dict, logits of last layer. {'pred_logits' and 'pred_boxes'}

            targets: List[Dict], len = Batchsize, which is a list of Dicts, contains coco-format ground truth

            matcher: A matcher that can match (targets, groundtruth) and (outputs, prediction).
            
            print_imgs: Boolean, if true save all images with plotted bounding boxes, including all layers' predictions and label

        Usage Guidence:
            to iterate all layer's outputs:
                `for i, output in enumerate(all_outputs):` 
                output['pred_logits'].shape  is  B x Q x C
                output['pred_boxes'].shape   is  B x Q x 4
        
        '''

        print("inspecting batch_query.......")
        # get all outputs.
        all_outputs = auxoutputs + [finaloutputs]
        # print(type(targets))
        # print(len(targets))
        # print(targets)
        print(len(all_outputs))
        print()
        if output_dir != None:
            self.out_dir = output_dir

        B = finaloutputs['pred_logits'].shape[0]

        # iterate all images in a batch
        for imgidx in range(B):
            image_id = targets[imgidx]['image_id'].item()
            image_id = str(image_id)[:4] + "_" + str(image_id)[4:]
            print(f'----------IMG_{image_id}----------')
            confidence_record_dict = {}
            flag_record = []
            n_targets = targets[imgidx]['labels'].shape[0]
            # print("n_targets", n_targets)
            n_targets = min(n_targets, 25)
            iou_record = []
            record = []
            # iterate all layers
            # print("imgidx", imgidx)
            target_labels = targets[imgidx]['labels']
            # target_bbx = targets[imgidx]['boxes']
            targets_img_i = targets[imgidx]
            # print("n_targets", n_targets)
            for i in range(n_targets):
                confidence_record_dict[i] = {}
                confidence_record_dict[i]['confidence'] = []
                confidence_record_dict[i]['iou'] = []
                confidence_record_dict[i]['label'] = []
            for lidx, output in enumerate(all_outputs):
                # print("lidx: ", lidx)
                query_idx, target_idx = matcher(output, targets)[imgidx]
                src_logits = output['pred_logits'][imgidx].unsqueeze(0)
                src_boxes = output['pred_boxes'][imgidx].unsqueeze(0)
                # print("src_boxes.shape")
                # print(src_boxes[0, query_idx, :].shape)
                # print(src_logits.shape)
                for i in range(n_targets):
                    # print(query_idx[i])
                    pred_confidence = max(torch.softmax(src_logits, dim=2)[0, query_idx[i], :])
                    pred_label = torch.argmax(torch.softmax(src_logits, dim=2)[0, query_idx[i], :])
                    # print(pred_confidence.item())
                    flag = pred_label.item() == target_labels[target_idx[i]].item()
                    # print(int(flag))
                    confidence_record_dict[i]['confidence'].append([pred_confidence.item()])
                    confidence_record_dict[i]['label'].append([flag, pred_label.item(), target_labels[target_idx[i]].item()])
                    flag_record.append([flag])
                    pred_bbx = src_boxes[0, query_idx[i], :]
                    target_bbx = targets_img_i['boxes'][target_idx[i]]
                    # print("pred_bbx", pred_bbx)
                    # print("target_bbx", target_bbx)
                    iou_i = iou(pred_bbx, target_bbx).item()
                    confidence_record_dict[i]['iou'].append([iou_i])
                    # print("pred_label: ", pred_label)
                if print_imgs:
                    # print(str(image_id.item()))
                    # print(str(image_id))
                    # print(str(image_id)[:4])
                    # print(str(image_id)[4:])
                    # print(image_id)
                    # image_path = "data/COCO/smalltest/val2017/JPEGImages/" + str(image_id) + ".jpg"
                    image_path = "data/COCO/test/val2017/JPEGImages/" + str(image_id) + ".jpg"
                    save_dir = self.out_dir + f"plotted_imgs/test_total/layer_{lidx}/"
                    # save_dir = self.out_dir + f"plotted_imgs/smalltest_total/layer_{lidx}/"
                    save_name = str(image_id) + ".jpg"
                    # img_path = 
                    # print(img_path)
                    predictions = {}
                    target_idx = target_idx.to(src_boxes.device)
                    predictions['boxes'] = tensor_to_numpy(torch.index_select(src_boxes[0, query_idx, :], 0, target_idx))
                    pred_targets = torch.argmax(torch.softmax(src_logits, dim=2)[0, query_idx, :], dim=1)
                    predictions['labels'] = tensor_to_numpy(torch.index_select(pred_targets, 0, target_idx))
                    img_i_targets = {}
                    img_i_targets['boxes'] = tensor_to_numpy(targets_img_i['boxes'])
                    img_i_targets['labels'] = tensor_to_numpy(targets_img_i['labels'])
                    # print(predictions)
                    # print(img_i_targets)
                    plot_bounding_boxes(image_path, predictions, img_i_targets, save_dir, save_name)
                    # print(predictions['label'])
                    # predictions['boxes']

                    # targets_img_i = 
                    # print(targets_img_i)
                    # print(predictions)

                    # print(predictions)
                    # print(query_idx)
                    # print(query_idx.shape)
                    # print(src_logits.shape)
                    # print(torch.softmax(src_logits, dim=2)[0, query_idx, :].shape)
                    # print(torch.argmax(torch.softmax(src_logits, dim=2)[0, query_idx, :], dim=1).shape)
                    # print(torch.argmax(torch.softmax(src_logits, dim=2)[0, query_idx, :], dim=1))
                    # pred_label = torch.argmax(torch.softmax(src_logits, dim=2)[0, query_idx :])
                    # print()
                    # predictions['label'] = tensor_to_numpy(torch.index_select(src_boxes[0, query_idx, :], 0, target_idx)))
                # score = src_logits[0, 0, :]
                # print(score)
                # print(query_idx, target_idx)
                # pred_bbx = src_boxes[query_idx[imgidx]]
                # target_bbx = targets[imgidx]['boxes']
                # print("src_logits.shape = ", src_logits.shape)
                # print("src_boxes.shape = ", src_boxes.shape)
                # `indices` is used for recording the closest query to the ground truth
                # indices = matcher(output, targets)[imgidx]

            for i in range(n_targets):
                confidence_record_dict[i]['confidence'].append([image_id])
            # print(confidence_record_dict[i]['confidence'])
            for i in range(n_targets):
                self.records.extend(list(zip(*list(confidence_record_dict[i]['confidence']))))
                self.records.extend(list(zip(*list(confidence_record_dict[i]['iou']))))
                self.records.extend(list(zip(*list(confidence_record_dict[i]['label']))))

            # print(confidence_record)
            # self.records.extend(confidence_record)

    def get_TP_scores(self, auxoutputs: List[Dict], finaloutputs, targets, matcher):
        
        ''' Inspect a batch of data, for specific use.

        Inputs:
            auxoutputs: List[Dict], len = Batchsize, which is a list of Dicts, 
                contains logits of each layer. {'pred_logits' and 'pred_boxes'}

            finaloutputs: Dict, logits of last layer. {'pred_logits' and 'pred_boxes'}

            targets: List[Dict], len = Batchsize, which is a list of Dicts, contains coco-format ground truth

            matcher: A matcher that can match (targets, groundtruth) and (outputs, prediction).
            
            print_imgs: Boolean, if true save all images with plotted bounding boxes, including all layers' predictions and label

        Usage Guidence:
            to iterate all layer's outputs:
                `for i, output in enumerate(all_outputs):` 
                output['pred_logits'].shape  is  B x Q x C
                output['pred_boxes'].shape   is  B x Q x 4
        
        '''
        all_outputs = auxoutputs + [finaloutputs]
        B = finaloutputs['pred_logits'].shape[0]
        # all_records = {}
        # all_records[]


        # iterate all images in a batch
        for imgidx in range(B):
            image_id = targets[imgidx]['image_id'].item()
            image_id = str(image_id)[:4] + "_" + str(image_id)[4:]
            print(f'----------IMG_{image_id}----------')
            confidence_record_dict = {}
            flag_record = []
            n_targets = targets[imgidx]['labels'].shape[0]
            n_targets = min(n_targets, 25)
            # iterate all layers
            # print("imgidx", imgidx)
            target_labels = targets[imgidx]['labels']
            # target_bbx = targets[imgidx]['boxes']
            targets_img_i = targets[imgidx]
            # print("n_targets", n_targets)
            for i in range(n_targets):
                confidence_record_dict[i] = {}
                confidence_record_dict[i]['confidence'] = []
                confidence_record_dict[i]['label'] = []
                confidence_record_dict[i]['iou'] = []
            for lidx, output in enumerate(all_outputs):
                # print("lidx: ", lidx)
                query_idx, target_idx = matcher(output, targets)[imgidx]
                src_logits = output['pred_logits'][imgidx].unsqueeze(0)
                src_boxes = output['pred_boxes'][imgidx].unsqueeze(0)
                # print("src_boxes.shape")
                # print(src_boxes[0, query_idx, :].shape)
                # print(src_logits.shape)
                for i in range(n_targets):
                    # print(query_idx[i])
                    pred_confidence = max(torch.softmax(src_logits, dim=2)[0, query_idx[i], :])
                    pred_label = torch.argmax(torch.softmax(src_logits, dim=2)[0, query_idx[i], :])
                    # print(pred_confidence.item())
                    flag = pred_label.item() == target_labels[target_idx[i]].item()
                    # print(int(flag))
                    confidence_record_dict[i]['confidence'].append([pred_confidence.item()])
                    confidence_record_dict[i]['label'].append([flag, pred_label.item(), target_labels[target_idx[i]].item()])
                    flag_record.append([flag])
                    pred_bbx = src_boxes[0, query_idx[i], :]
                    target_bbx = targets_img_i['boxes'][target_idx[i]]
                    # print("pred_bbx", pred_bbx)
                    # print("target_bbx", target_bbx)
                    iou_i = iou(pred_bbx, target_bbx).item()
                    confidence_record_dict[i]['iou'].append(iou_i)

            # Integrating scores
            for i in range(n_targets):
                if confidence_record_dict[i]['label'][-1][0] == True:
                    tp = [False, False]
                    if confidence_record_dict[i]['iou'][-1] >= 0.5:
                        self.n_tp[0] += 1
                        tp[0] = True
                    if confidence_record_dict[i]['iou'][-1] >= 0.75:
                        self.n_tp[1] += 1
                        tp[1] = True
                    fp_e = [False, False]
                    for lidx in range(len(all_outputs)-1):
                        if confidence_record_dict[i]['label'][lidx][0] == True:
                            if confidence_record_dict[i]['iou'][lidx] >= confidence_record_dict[i]['iou'][-1] \
                                and confidence_record_dict[i]['confidence'][lidx][0] >= confidence_record_dict[i]['confidence'][-1][0]:
                                if tp[0]:
                                    self.n_tp_f[0] += 1
                                if tp[1]:
                                    self.n_tp_f[1] += 1
                                break

                if confidence_record_dict[i]['label'][-1][0] == False:
                    fp = [False, False]
                    if confidence_record_dict[i]['iou'][-1] >= 0.5:
                        self.n_fp[0] += 1
                        fp[0] = True
                    if confidence_record_dict[i]['iou'][-1] >= 0.75:
                        self.n_fp[1] += 1
                        fp[1] = True
                    if not fp[0]:
                        break
                    for lidx in range(len(all_outputs)-1):
                        if confidence_record_dict[i]['label'][lidx][0] == False:
                            if confidence_record_dict[i]['confidence'][lidx][0] <= confidence_record_dict[i]['confidence'][-1][0]:
                                if confidence_record_dict[i]['iou'][lidx] >= 0.5:
                                    if fp[0]:
                                        self.n_fp_e[0] += 1
                                if confidence_record_dict[i]['iou'][lidx] >= 0.75:
                                    if fp[1]:
                                        self.n_fp_e[1] += 1
                                break
        # print(self.n_tp)
        # print(self.n_fp)
        # print(self.n_tp_f)
        # print(self.n_fp_e)
        try:
            print("IoU>0.50")
            print("TP F Rate: ", self.n_tp_f[0]/self.n_tp[0])
            print("FP E Rate: ", self.n_fp_e[0]/self.n_fp[0])
            print("IoU>0.75")
            print("TP F Rate: ", self.n_tp_f[1]/self.n_tp[1])
            print("FP E Rate: ", self.n_fp_e[1]/self.n_fp[1])
        except:
            print("Accumulating stage...")



            # for lidx in range(6):
            #     predictions = {}
            #     target_idx = target_idx.to(src_boxes.device)
            #     predictions['boxes'] = tensor_to_numpy(torch.index_select(src_boxes[0, query_idx, :], 0, target_idx))
            #     pred_targets = torch.argmax(torch.softmax(src_logits, dim=2)[0, query_idx, :], dim=1)
            #     predictions['labels'] = tensor_to_numpy(torch.index_select(pred_targets, 0, target_idx))
            #     img_i_targets = {}
            #     img_i_targets['boxes'] = tensor_to_numpy(targets_img_i['boxes'])
            #     img_i_targets['labels'] = tensor_to_numpy(targets_img_i['labels'])




