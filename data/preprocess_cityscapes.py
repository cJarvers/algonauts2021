'''
Wrapper / loader for Cityscapes dataset.
'''
import os
import math
import datetime
from typing import Dict
import numpy as np
import random
import pathlib
from pathlib import Path
import torch
import torchvision
from torchvision.transforms import ConvertImageDtype, Resize, Compose, Lambda, Normalize
from torch.utils.data import Dataset
from PIL import Image
from data.utils import subsample_ids
from data.cityscapes_loader import CityscapesDB

def tolong(x):
    return(x.long())
    
def squeeze(x):
    return(x.squeeze())

class CityscapesPreprocessor:
    '''
    Wrapper for the Cityscapes dataset.
    Args:
        *root_dir (str): Directory from which to load data. Should contain
                         subfolders `Annotations`, `ImageSets`, and `JPEGImages`.
        *phase (str): `training` or `validation`
        *nframes (int): number of frames to subsample every video to
        *transform: PyTorch transform to apply to the video frames
        *label_transform: PyTorch transform to apply to the label frames
        *common_transform: PyTorch transform to apply to both the images and the labels
        *rng_state (int): initial state of the random number generator
    '''
    def __init__(self, root_dir, phase, nframes, transform=None, label_transform=None, common_transform=None, rng_state=0):
        self.root_dir = root_dir
        self.phase = phase
        self.nframes = nframes
        self.transform = transform
        self.label_transform = label_transform
        self.common_transform = common_transform
        self.dataset = CityscapesDB(root_dir, phase, nframes, rng_state=rng_state)

    def convert(self, idx):
        (city_key, seq_key) = self.dataset.get_ids()[idx]
        (vid, ann) = self.dataset.get_sample(idx)

        if self.common_transform:
            vid = self.common_transform(vid)
            ann = self.common_transform(ann)
        if self.transform:
            vid = self.transform(vid)
        if self.label_transform:
            ann = self.label_transform(ann)
        
        if self.phase == 'training':
            path = os.path.join(self.root_dir, )
            torch.save(vid, path)
        elif self.phase = 'validation':
        
        else
            raise ValueError(f'Encountered unknown phase: {self.phase}')


    def __len__(self):
       return len(self.dataset.get_ids())
       
if __name__ == '__main__':
    transform = Compose([ConvertImageDtype(torch.float32), Resize((224, 224))])
    label_transform = Compose([Lambda(tolong), Resize((224, 224)), Lambda(squeeze)])
    p = CityscapesPreprocessor('/data/cityscapes', 'training', 16, transform, label_transform)
    
    print(f'Preprocessing {len(p)} training videos. Time: {datetime.datetime.now()}', flush=True)
    for i in range(len(p)):
        p.convert(i)
        if (i+1) % 100 == 0:
            print(f'Processed {i+1} videos. Time: {datetime.datetime.now()}', flush=True)
    
    p = CityscapesPreprocessor('/data/cityscapes', 'validation', 16, transform, label_transform)
    
    print(f'Preprocessing {len(p)} validation videos. Time: {datetime.datetime.now()}', flush=True)
    for i in range(len(p)):
        p.convert(i)
        if (i+1) % 100 == 0:
            print(f'Processed {i+1} videos. Time: {datetime.datetime.now()}', flush=True)
