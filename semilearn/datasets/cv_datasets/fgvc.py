# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import json
import torchvision
import numpy as np
import math
from PIL import Image

from torchvision import transforms, datasets
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


def get_fgvc(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    data_dir = os.path.join(data_dir, name.lower())
    
    variant_file = os.path.join(data_dir, "variants.txt")
    train_file = os.path.join(data_dir, "images_variant_trainval.txt")
    val_file = os.path.join(data_dir, "images_variant_val.txt")
    test_file = os.path.join(data_dir, "images_variant_test.txt")
    image_dir = os.path.join(data_dir, "images")

    cname2lab = read_fgvcaircraft_classes(variant_file)

    train_data, train_targets = read_fgvcaircraft_data(train_file, image_dir, cname2lab)
    val_data, val_targets = read_fgvcaircraft_data(val_file, image_dir, cname2lab)
    test_data, test_targets = read_fgvcaircraft_data(test_file, image_dir, cname2lab)

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_extreme = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(5, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])
    
    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, train_data, train_targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)

    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    if alg == 'fullysupervised':
        lb_data = train_data
        lb_targets = train_targets

    lb_dset = FGVCAircraftDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, None, False)
    ulb_dset = FGVCAircraftDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_extreme, transform_strong, False)

    eval_dset = FGVCAircraftDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, None, False)

    return lb_dset, ulb_dset, eval_dset


def read_fgvcaircraft_classes(variant_file):

    with open(variant_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    cname2lab = {name: i for i, name in enumerate(class_names)}
    return cname2lab


def read_fgvcaircraft_data(split_file, image_dir, cname2lab):

    items = []

    with open(split_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            imname = parts[0] + ".jpg"
            classname = ' '.join(parts[1:])
            
            if classname not in cname2lab:
                print(f"Warning: '{classname}' not found in cname2lab!")
                continue
            
            label = cname2lab[classname]
            impath = os.path.join(image_dir, imname)

            if not os.path.exists(impath):
                print(f"Warning: File {impath} does not exist!")
                continue

            items.append((impath, label))

    return [item[0] for item in items], [item[1] for item in items]


class FGVCAircraftDataset(BasicDataset):
    def __sample__(self, idx):

        path = self.data[idx]
        img = Image.open(path).convert("RGB")
        target = self.targets[idx]
        return img, target
