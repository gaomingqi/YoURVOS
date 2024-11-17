'''
Inference code for OMFormer, on Ref-Youtube-VOS
Modified from ReferFormer (https://github.com/wjn922/ReferFormer)
'''
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F
import json

import opts
from tqdm import tqdm

import multiprocessing as mp
import threading

from tools.colormap import colormap


# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()


# build transform
transform = T.Compose([
	T.Resize(360),
	T.ToTensor(),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main(args):
	args.masks = True
	args.batch_size == 1
	print("Inference only supports for batch size = 1") 

	# fix the seed for reproducibility
	seed = args.seed + utils.get_rank()
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	split = args.split
	# save path
	output_dir = args.output_dir
	save_path_prefix = os.path.join(output_dir, split)
	if not os.path.exists(save_path_prefix):
		os.makedirs(save_path_prefix)

	save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
	if args.visualize:
		if not os.path.exists(save_visualize_path_prefix):
			os.makedirs(save_visualize_path_prefix)

	# load data
	root = Path(args.yourvos_path)
	img_folder = os.path.join(root, split, "JPEGImages")
	meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
	with open(meta_file, "r") as f:
		data = json.load(f)["videos"]
	valid_videos = set(data.keys())
	video_list = sorted([video for video in valid_videos])

	# create subprocess
	thread_num = args.ngpu
	global result_dict
	result_dict = mp.Manager().dict()

	processes = []
	lock = threading.Lock()

	video_num = len(video_list)
	per_thread_video_num = video_num // thread_num

	start_time = time.time()
	print('Start inference')


	sub_processor(args, data, save_path_prefix, save_visualize_path_prefix, img_folder, video_list)


	end_time = time.time()
	total_time = end_time - start_time

	result_dict = dict(result_dict)
	num_all_frames_gpus = 0
	for pid, num_all_frames in result_dict.items():
		num_all_frames_gpus += num_all_frames

	print("Total inference time: %.4f s" %(total_time))

def sub_processor(args, data, save_path_prefix, save_visualize_path_prefix, img_folder, video_list):
	
	pid = 0

	# model
	model, criterion, _ = build_model(args) 
	device = args.device
	model.to(device)

	model_without_ddp = model
	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

	if pid == 0:
		print('number of params:', n_parameters)

	if args.resume:
		checkpoint = torch.load(args.resume, map_location='cpu')
		missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
		unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
		if len(missing_keys) > 0:
			print('Missing Keys: {}'.format(missing_keys))
		if len(unexpected_keys) > 0:
			print('Unexpected Keys: {}'.format(unexpected_keys))
	else:
		raise ValueError('Please specify the checkpoint for inference.')


	# start inference
	num_all_frames = 0 
	model.eval()

	# 1. For each video
	for video in tqdm(video_list):
		metas = [] # list[dict], length is number of expressions

		expressions = data[video]["expressions"]   
		expression_list = list(expressions.keys()) 
		num_expressions = len(expression_list)
		video_len = len(data[video]["frames"])

		# read all the anno meta
		for i in range(num_expressions):
			meta = {}
			meta["video"] = video
			meta["exp"] = expressions[expression_list[i]]["exp"]
			meta["exp_id"] = expression_list[i]
			meta["frames"] = data[video]["frames"]
			metas.append(meta)
		meta = metas

		# 2. For each expression
		for i in range(num_expressions):
			video_name = meta[i]["video"]
			exp = meta[i]["exp"]
			exp_id = meta[i]["exp_id"]
			frames = meta[i]["frames"]

			video_len = len(frames)
			# store images
			imgs = []
			for t in range(video_len):
				frame = frames[t]
				img_path = os.path.join(img_folder, video_name, frame + ".jpg")
				img = Image.open(img_path).convert('RGB')
				origin_w, origin_h = img.size
				imgs.append(transform(img)) # list[img]

			imgs = torch.stack(imgs, dim=0).to(args.device) # [video_len, 3, h, w]
			img_h, img_w = imgs.shape[-2:]
			size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
			target = {"size": size}

			with torch.no_grad():
				# spatial operations
				# split videos into short segments, encode, fuse, mask generation
				batch_len = 30
				img_sets = [imgs[(sid*batch_len):min(len(imgs), (sid+1)*batch_len)] for sid in range(0, len(imgs)//batch_len+1)]
				if len(imgs) % batch_len == 0:
					img_sets = img_sets[:-1]
				mask_list = []
				hs_list = []
				for sub_set in img_sets:
					# mask: 1, len_seq, 5, h, w
					# hs:   4, len_seq, 5, c
					mask_seq, hs_seq, text_features, text_sentence_features = model.spatial_inference([sub_set], [exp], [target])
					mask_list.append(mask_seq)
					hs_list.append(hs_seq)
					# outputs = model.spatial_inference([sub_set], [exp], [target])
				spatial_masks = torch.concat(mask_list, 1)	# 1, num_frames, 5, h, w
				spatial_hs = torch.concat(hs_list, 1)		# 4, num_frames, 5, c

				# temporal operations (over spatial_hs, object-centric features)
				pred_masks = model.temporal_inference(spatial_masks, spatial_hs, text_features, text_sentence_features)
			
			# pred_logits = outputs["pred_logits"][0] 
			# pred_boxes = outputs["pred_boxes"][0]   
			pred_masks = pred_masks[0]		# t, h, w
 
			pred_masks = pred_masks.unsqueeze(0)

			pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
			pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy() 

			all_pred_masks = pred_masks

			# save binary image
			save_path = os.path.join(save_path_prefix, video_name, exp_id)
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			for j in range(video_len):
				frame_name = frames[j]
				mask = all_pred_masks[j].astype(np.float32) 
				mask = Image.fromarray(mask * 255).convert('L')
				save_file = os.path.join(save_path, frame_name + ".png")
				mask.save(save_file)


# visuaize functions
def box_cxcywh_to_xyxy(x):
	x_c, y_c, w, h = x.unbind(1)
	b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
		 (x_c + 0.5 * w), (y_c + 0.5 * h)]
	return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
	img_w, img_h = size
	b = box_cxcywh_to_xyxy(out_bbox)
	b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
	return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
	W, H = img_size
	for i, ref_point in enumerate(reference_points):
		init_x, init_y = ref_point
		x, y = W * init_x, H * init_y
		cur_color = color
		draw.line((x-10, y, x+10, y), tuple(cur_color), width=4)
		draw.line((x, y-10, x, y+10), tuple(cur_color), width=4)

def draw_sample_points(draw, sample_points, img_size, color_list):
	alpha = 255
	for i, samples in enumerate(sample_points):
		for sample in samples:
			x, y = sample
			cur_color = color_list[i % len(color_list)][::-1]
			cur_color += [alpha]
			draw.ellipse((x-2, y-2, x+2, y+2), 
							fill=tuple(cur_color), outline=tuple(cur_color), width=1)

def vis_add_mask(img, mask, color):
	origin_img = np.asarray(img.convert('RGB')).copy()
	color = np.array(color)

	mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
	mask = mask > 0.5

	origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
	origin_img = Image.fromarray(origin_img)
	return origin_img

  

if __name__ == '__main__':
	parser = argparse.ArgumentParser('ReferFormer inference script', parents=[opts.get_args_parser()])
	args = parser.parse_args()
	main(args)
