# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

from torch.utils.data import ConcatDataset
from .refexp2seq import build as build_seq_refexp
from .refexp import build as build_refexp


def build(image_set, args):
    concat_data = []

    print('preparing coco2seq dataset ....')
    coco_names = ["refcoco", "refcoco+", "refcocog"]
    for name in coco_names:
        # coco_seq =  build_seq_refexp(name, image_set, args)
        coco_seq =  build_refexp(name, image_set, args)
        concat_data.append(coco_seq)

    print('preparing ytvos dataset  .... ')
    ytvos_dataset = build_ytvs(image_set, args)
    concat_data.append(ytvos_dataset)

    concat_data = ConcatDataset(concat_data)

    return concat_data