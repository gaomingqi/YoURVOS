"""
YoURVOS data loader
"""
from pathlib import Path

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random

from os import listdir  # for debug


class YoURVOSDataset(Dataset):
    """
    A dataset class for the YoURVOS dataset, first introducted in the paper:
    "Show Me When and Where: Towards Referring Video Object Segmentation in the Wild"
    (under review). 
    """
    def __init__(self, img_folder: Path, ann_file: Path, transforms, num_frames_short: int, 
                 num_frames_long: int, max_skip: int):
        """
        num_frames_short: number of frames for mask prediction
        num_frames_long: number of frames for long video prediction
        """
        self.img_folder = img_folder     
        self.ann_file = ann_file         
        self._transforms = transforms    
        self.num_frames_short = num_frames_short   
        self.max_skip = max_skip
        self.num_frames_long = num_frames_long
        # create video meta data
        self.prepare_metas()       

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        print('\n')    

    def prepare_metas(self):
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            for exp_id, exp_dict in vid_data['expressions'].items():
                span_list = exp_dict['span']
                for si, span in enumerate(span_list):
                    span_item = [span[0]//5, span[1]//5]    # convert to normailised index
                    step = (span_item[1] - span_item[0] + 1) // 10   # 10 samples per expression

                    if step == 0:
                        step = (span_item[1] - span_item[0] + 1) // 5   # 5 samples per expression
                        if step == 0:
                            continue

                    lw_bound = -1 if si == 0 else span_list[si-1][1]//5
                    up_bound = len(vid_frames)-1 if si == len(span_list)-1 else span_list[si+1][0]//5

                    for frame_id in range(span_item[0], span_item[1], step):
                        meta = {}
                        meta['video'] = vid
                        meta['exp'] = exp_dict['exp']
                        meta['exp_id'] = exp_id
                        meta['obj_id'] = int(exp_dict['obj_id'])
                        meta['frames'] = vid_frames
                        meta['frame_id'] = frame_id
                        meta['span'] = [span_item[0], span_item[1]-1]
                        meta['lw_bound'] = lw_bound
                        meta['up_bound'] = up_bound
                        self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        """
        sample one clip from the video (exclude other clips referred to by the SAME expression)
        each data sample consists of:
        - [num_frames] video frames (25 by default)
        - [num_masks] video masks (5 by default)

        one video:
        [tif], [trf], [trf], [trf], "[tif], [tif], [tif], [trf], [trf], [tif], [tif]"
        one sample:
        "[tif], [tif], [tif], [trf], [trf], [tif], [tif]"
        """
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, exp_id, obj_id, frames, frame_id, span, up_bound, lw_bound = \
                        meta['video'], meta['exp'], meta['exp_id'], meta['obj_id'], meta['frames'], meta['frame_id'], meta['span'], meta['up_bound'], meta['lw_bound']

            # clean up the caption
            exp = " ".join(exp.lower().split())
            vid_len = len(frames)
            num_frames_long = self.num_frames_long    # frames for mask generation
            num_frames_short = self.num_frames_short
            
            # random sparse sample
            sample_indx_m = [frame_id]
            if self.num_frames_long != 1:
                # local sample
                sample_id_before = random.randint(1, 10)
                sample_id_after = random.randint(1, 10)
                local_indx = [max(span[0], frame_id - sample_id_before), min(span[1], frame_id + sample_id_after)]
                sample_indx_m.extend(local_indx)
    
                # global sampling
                if num_frames_long > 3:
                    all_inds = list(range(vid_len))
                    # two rounds
                    # # 1. within span ()
                    # global_inds = all_inds[span[0]:min(sample_indx_m)] + all_inds[max(sample_indx_m):(span[1])]
                    # global_n = num_frames_long - len(sample_indx_m)     # 2 masks
                    # if len(global_inds) > global_n:
                    #     select_id = random.sample(range(len(global_inds)), global_n)
                    #     for s_id in select_id:
                    #         sample_indx_m.append(global_inds[s_id])
                    # elif (span[1]-span[0]) >=global_n:  # sample long range global frames
                    #     select_id = random.sample(range(span[0], span[1]), global_n)
                    #     for s_id in select_id:
                    #         sample_indx_m.append(all_inds[s_id])
                    # else:
                    #     select_id = random.sample(range(span[0], span[1]), global_n - (span[1]-span[0]+1)) + list(range(span[0], span[1]))           
                    #     for s_id in select_id:                                                                   
                    #         sample_indx_m.append(all_inds[s_id])

                    # 2. within all clip
                    global_inds = all_inds[(lw_bound+1):min(sample_indx_m)] + all_inds[max(sample_indx_m):(up_bound-1)]
                    global_n = num_frames_long - len(sample_indx_m)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx_m.append(global_inds[s_id])
                    elif (up_bound-lw_bound-1) >=global_n:  # sample long range global frames
                        select_id = random.sample(range(lw_bound+1, up_bound), global_n)
                        for s_id in select_id:
                            sample_indx_m.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(lw_bound+1, up_bound), global_n - (up_bound-lw_bound-1)) + list(range(lw_bound+1, up_bound))           
                        for s_id in select_id:                                                                   
                            sample_indx_m.append(all_inds[s_id])
            sample_indx_m.sort()

            # read frames and masks
            imgs, boxes, masks, valid = [], [], [], []
            # intersection between frame and mask indices
            mask_id_list = []

            # get mask size
            try:
                mask_pad = Image.open(os.path.join(str(self.img_folder), 'Annotations', video, exp_id, frames[frame_id] + '.png')).convert('P')
                mask_pad = np.zeros_like(np.array(mask_pad))
                mask_pad = torch.from_numpy(mask_pad)   # zero mask for frames without a mask
            except:
                print(f'empty: {video}, {exp_id}, {frames[frame_id]} + .png')
                idx = random.randint(0, self.__len__() - 1)
                continue

            for j in range(self.num_frames_long):
                frame_indx = sample_indx_m[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                img = Image.open(img_path).convert('RGB')

                try:    # inside span
                    mask_path = os.path.join(str(self.img_folder), 'Annotations', video, exp_id, frame_name + '.png')
                    mask = Image.open(mask_path).convert('P')
                    mask = np.array(mask)
                    mask = (mask==obj_id).astype(np.float32) # 0,1 binary
                    if (mask > 0).any():
                        y1, y2, x1, x2 = self.bounding_box(mask)
                        box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                        valid.append(1)
                    else: # some frame didn't contain the instance
                        box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                        valid.append(0)
                    mask = torch.from_numpy(mask)

                    # append
                    masks.append(mask)
                    boxes.append(box)
                except: # outside span
                    masks.append(mask_pad)
                    valid.append(0)
                    boxes.append(torch.tensor([0, 0, 0, 0]).to(torch.float))

                imgs.append(img)

            # transform
            w, h = img.size
            boxes = torch.stack(boxes, dim=0) 
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0) 
            target = {
                'frames_idx': torch.tensor(sample_indx_m), # [T,]
                'boxes': boxes,                          # [T, 4], xyxy
                'masks': masks,                          # [T, H, W]
                'valid': torch.tensor(valid),            # [T,]
                'caption': exp,
                'orig_size': torch.as_tensor([int(h), int(w)]), 
                'size': torch.as_tensor([int(h), int(w)])
            }

            # # -------------------------------------------
            # # debug: before transformation
            # # clean first
            # temp_files = './debug'
            # for file_name in listdir(temp_files):
            #     if file_name.endswith('.png'):
            #         os.remove(f'{temp_files}/{file_name}')
            #     if file_name.endswith('.jpg'):      
            #         os.remove(f'{temp_files}/{file_name}')
            # for i, (image, mask) in enumerate(zip(imgs, target['masks'])):
            #     image.save(f'{temp_files}/{i:02d}_before.jpg')
            #     # Image.fromarray((np.uint8(mask.numpy())*255)[np.newaxis, :]).save(f'{temp_files}/{i:02d}_before.png')
            #     Image.fromarray(np.uint8(mask.numpy()*255)).convert('L').save(f'{temp_files}/{i:02d}_before.png')
            # # end of debug
            # # -------------------------------------------

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            imgs, target = self._transforms(imgs, target)   # imgs (Image), target['masks'], cpu

            # # -------------------------------------------
            # # debug: after transformation
            # # mean = [0.485, 0.456, 0.406]
            # # std = [0.229, 0.224, 0.225]
            # for i, (image, mask) in enumerate(zip(imgs, target['masks'])):
            #     image = image.numpy()
            #     image = image - np.min(image)
            #     image = np.uint8(image / np.max(image) * 255)
            #     Image.fromarray(image.transpose(1, 2, 0)).save(f'{temp_files}/{i:02d}_after.jpg')
            #     Image.fromarray(np.uint8(mask.numpy()*255)).convert('L').save(f'{temp_files}/{i:02d}_after.png')
            # # end of debug
            # # -------------------------------------------

            valid_index = [i for i, v in enumerate(target['valid']) if v == 1]
            if len(valid_index) == 0:
                continue
            
            target_span = [min(valid_index), max(valid_index)]
            
            # random select num_frames frames (with masks)
            try:
                mask_id_list = random.sample(valid_index, num_frames_short)
                mask_id_list.sort()
            except:
                pass

            # full targets
            target_full = {
                'frames_idx': target['frames_idx'],   # [num_frames_full]
                'boxes': target['boxes'],             # [num_frames_full, 4], xyxy
                'masks': target['masks'][mask_id_list],             # [num_frames_short, H, W]
                'valid': target['valid'],             # [num_frames_full,]
                'caption': target['caption'],
                'orig_size': target['orig_size'], 
                'size': target['size'],
                'span': target_span,
                'mask_id': mask_id_list,
            }

            imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
            
            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'][mask_id_list] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target_full


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            normalize,
        ])
    
    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.yourvos_path)
    assert root.exists(), f'provided YoURVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "meta_expressions" / "train" / "meta_expressions.json"),
        "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),    # not used actually
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = YoURVOSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size), num_frames_short=args.num_frames, 
                           num_frames_long=args.num_frames_long, max_skip=args.max_skip)
    return dataset
