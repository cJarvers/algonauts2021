#!/usr/bin/env python3
'''Perform multiGPU training of 3D-ResNet50 model on multiple datasets.'''
import torch
from data.moments_loader import MomentsDataset
from data.objectron_loader import ObjectronDataset
from models.resnet3d50 import ResNet3D50Backbone
from utils.training import multidata_train

if __name__ == '__main__':
    # parse command line arguments and check that environment is set up
    assert torch.cuda.is_available(), 'Script requires GPU, but cuda not available.'
    
    # set up model and decoders
    backbone = ResNet3D50Backbone()
    
    # load datasets
    mit = MomentsDataset('data/Moments_in_Time_Raw', 'training', 16)
    obj = ObjectronDataset('data/objectron', 16)
    
    # launch training
    # multidata_train(...)
