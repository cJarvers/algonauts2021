#!/usr/bin/env python3
'''Perform multiGPU training of 3D-ResNet50 model on multiple datasets.'''
# TODO:
# - Currently, frames are only converted to float and resized to 224x224.
#   We may want to add normalization and augmentations.
# - Only 2 datasets used so far.
# - Currently, we only use training data and log the losses. We may want to
#   also evaluate accuracy or something similar on validation data.
# - Currently, the datasets used are hardcoded. It may be nicer to set flags
#   via the commandline to activate / deactivate certain datasets.
# - Logging / checkpointing functionality not implemented yet.

# standard Python imports
import argparse
# PyTorch-related imports
import torch
import torch.multiprocessing as mp
from torchvision.transforms import ConvertImageDtype, Resize, Compose
from torch.utils.data import DataLoader
# our custom imports
from data.moments_loader import MomentsDataset
from data.objectron_loader import ObjectronDataset
from models.decoders import ClassDecoder
from models.resnet3d50 import ResNet3D50Backbone
from utils.training import multidata_train

# set up command line parsing
parser = argparse.ArgumentParser(description='Perform training of 3D-ResNet50 model on multiple datasets.')
parser.add_argument('--bsize', type=int, default=32, help='Batch size')
parser.add_argument('-d', '--devices', type=str, nargs='*', default=['cuda:0', 'cuda:1'], help='Names of devices to train on.')
parser.add_argument('-n', '--nprocs', type=int, default=2, help='Number of processes to launch.')
parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to train.')

def log_fn(epoch, loss, metric, model_dict, decoder_dict, rank):
    None

if __name__ == '__main__':
    # parse command line arguments and check that environment is set up
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'Script requires GPU, but cuda not available.'
    
    # set up model and decoders
    backbone = ResNet3D50Backbone
    moments_decoder = ClassDecoder(305)
    objectron_decoder = ClassDecoder(9)
    decoders = [moments_decoder, objectron_decoder]
    
    # load datasets
    transform = Compose([ConvertImageDtype(torch.float32), Resize((224, 224))])
    moments = MomentsDataset('data/Moments_in_Time_Raw', 'training', 16, transform=transform)
    moments_loader = DataLoader(moments, batch_size=args.bsize, shuffle=True)
    objectron = ObjectronDataset('data/objectron', 16, transform=transform)
    objectron_loader = DataLoader(objectron, batch_size=args.bsize, shuffle=True)
    datasets = [(moments_loader, []), (objectron_loader, [])]
    
    # set up remaining training infrastructure
    devices = (args.devices * len(datasets))[:len(datasets)] # if there are more dataset than devices, distribute
    moments_loss = torch.nn.CrossEntropyLoss().cuda()
    objectron_loss = torch.nn.CrossEntropyLoss().cuda()
    losses = [moments_loss, objectron_loss]
    metrics = [(), ()]
    
    # launch training
    n = args.nprocs
    assert n == len(datasets), 'Number of training processes does not match number of datasets.'
    mp.spawn(multidata_train,
         args=(n, backbone, datasets, decoders, losses, metrics, devices, log_fn, args.epochs, True),
         nprocs=n,
         join=True)
