import torch.utils.data
import torchvision

from .yourvos import build as build_yourvos
# from .refexp import build as build_refexp         # single image
from .concat_dataset import build as build_joint
from .refexp2seq import build as build_seq_refexp   # image sequence
from .refexp import build as build_refexp   # image sequence


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    # for main training on YoURVOS
    if dataset_file == 'yourvos':
        return build_yourvos(image_set, args)

    # for pretraining
    if dataset_file == "refcoco" or dataset_file == "refcoco+" or dataset_file == "refcocog":
        return build_refexp(dataset_file, image_set, args)
        # return build_seq_refexp(dataset_file, image_set, args)
    # for joint training of refcoco and ytvos
    if dataset_file == 'joint':
        return build_joint(image_set, args)
    raise ValueError(f'dataset {dataset_file} not supported')
