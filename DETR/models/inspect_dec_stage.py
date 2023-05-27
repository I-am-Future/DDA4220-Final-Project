import torch
from typing import List, Dict
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import torch.nn.functional as F
import pickle

def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx





class MyInspector():
    def __init__(self) -> None:
        self.records = []

    def dump(self):
        ''' Dump the all statistics. '''
        # print(self.records)
        with open('work_dir_dev/stat1.pkl', 'wb') as f:
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
                # looks like: tensor([7, 2], device='cuda:0'), 7, 2 is correct answer
                target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targetimg, indicesimg)])

                # then we can calulate the acc by query logits and target
                query_logits = src_logits[idx] # size [2, 22]
                print(query_logits.argmax(axis=1), target_classes_o)
                print(f'stage {lidx} acc:', (accuracy(src_logits[idx], target_classes_o)[0]).cpu().item()  )

                ### investigating on bbox

                ## Get the target boxes
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targetimg, indicesimg)], dim=0)
                query_boxes = src_boxes[idx]
                loss_bbox = F.l1_loss(query_boxes, target_boxes, reduction='none')
                print(f'stage {lidx} loss:', loss_bbox.sum(axis=1)  )


    def inspect_batch_query(self, auxoutputs: List[Dict], finaloutputs, targets, matcher):
        ''' Inspect a batch of data, for specific use.

        Inputs:
            auxoutputs: List[Dict], len = Number of Layers, which is a list of Dicts, 
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
            # print('----------IMG----------')
            record = []
            # iterate all layers
            for lidx, output in enumerate(all_outputs):
                src_logits = output['pred_logits'][imgidx].unsqueeze(0)
                src_boxes = output['pred_boxes'][imgidx].unsqueeze(0)

                # `indices` is used for recording the closest query to the ground truth
                indices = matcher(output, targets)[imgidx][0]
                # print(indices)
                record.append(indices.tolist())
            self.records.extend(list(zip(*record)))

