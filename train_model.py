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

# standard Python imports
import argparse
from functools import partial
# PyTorch-related imports
import torch
import torch.multiprocessing as mp
from torchvision.transforms import ConvertImageDtype, Resize, Compose
from torch.utils.data import DataLoader
# our custom imports
from data.moments_loader import MomentsDataset
from data.objectron_loader import ObjectronDataset
from data.youtube_faces_loader import YouTubeFacesDataset
from data.davis_loader import DAVISDataset
from models.decoders import make_moments_decoder, make_objectron_decoder, make_youtube_faces_decoder, make_davis_decoder
from models.resnet3d50 import ResNet3D50Backbone
from utils.training import multidata_train
from utils.utils import Logger
from utils.losses import NT_Xent

# set up command line parsing
parser = argparse.ArgumentParser(description='Perform training of 3D-ResNet50 model on multiple datasets.')
parser.add_argument('--bsize', type=int, default=32, help='Batch size')
parser.add_argument('-d', '--devices', type=str, nargs='*', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], help='Names of devices to train on.')
parser.add_argument('-n', '--nprocs', type=int, default=3, help='Number of processes to launch.')
parser.add_argument('-b', '--batches', type=int, default=1000, help='Number of batches to train.')
parser.add_argument('--logpath', type=str, default='/mnt/logs/', help='Path to save log files to.')
parser.add_argument('--ckptpath', type=str, default='/mnt/logs/', help='Path to save checkpoints to.')
parser.add_argument('--loginterval', type=int, default=1000, help='Number of batches after which to perform validation and log losses & metrics.')
parser.add_argument('--ckptinterval', type=int, default=1000, help='Number of batches after which to save logs and checkpoints (should be a multiple of loginterval).')
parser.add_argument('--resume', dest='resume', action='store_true', help='Resume training from pervious checkpoint.')
parser.set_defaults(resume=False)


if __name__ == '__main__':
    # parse command line arguments and check that environment is set up
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'Script requires GPU, but cuda not available.'

    # set up model and decoders
    backbone = ResNet3D50Backbone
    if args.resume: # to resume previous training, load weights from previous checkpoint
        weights = torch.load(f'model_0_b{batchnum}.ckpt')
        backbone.load_state_dict(weights)
        log = torch.load(logpath + 'rank0.log')
        batchnum = log['loss'][-1][1]
        moments_weights = torch.load(f'decoder_0_b{batchnum}.ckpt')
        moments_decoder = partial(make_moments_decoder, weights=moments_weights)
        objectron_weights = torch.load(f'decoder_1_b{batchnum}.ckpt')
        objectron_decoder = partial(make_objectron_decoder, weights=objectron_weights)
        youtube_faces_weights = torch.load(f'decoder_2_b{batchnum}.ckpt')
        youtube_faces_decoder = partial(make_youtube_faces_decoder, weights=youtube_faces_weights)
        davis_weights = torch.load(f'decoder_3_b{batchnum}.ckpt')
        davis_decoder = partial(make_davis_decoder, weights=davis_weights)
    else:
        moments_decoder = make_moments_decoder
        objectron_decoder = make_objectron_decoder
        youtube_faces_decoder = make_youtube_faces_decoder
        davis_decoder = make_davis_decoder
    decoders = [moments_decoder, objectron_decoder, youtube_faces_decoder, davis_decoder]

    # load datasets
    transform = Compose([ConvertImageDtype(torch.float32), Resize((224, 224))])
    moments = MomentsDataset('/data/Moments_in_Time_Raw', 'training', 16, transform=transform)
    moments_loader = DataLoader(moments, batch_size=args.bsize, shuffle=True, num_workers=5)
    objectron = ObjectronDataset('/data/objectron', 16, transform=transform)
    objectron_loader = DataLoader(objectron, batch_size=args.bsize, shuffle=True, num_workers=5)
    yt_faces = YouTubeFacesDataset('/data/YouTubeFaces', 'training', 16, transform=transform)
    # shuffling for yt_faces happens within the dataset implementation
    # as it's an iterable dataset
    yt_faces_loader = DataLoader(yt_faces, batch_size=args.bsize, shuffle=False, drop_last=True,
                                 worker_init_fn=YouTubeFacesDataset.worker_init_fn, num_workers=5)
    davis = DAVISDataset('/data/DAVIS', 'training', 16, transform, lambda x: x)
    davis_loader = DataLoader(davis, batch_size=4, shuffle=True, drop_last=True, num_workers=4)
    datasets = [(moments_loader, []), (objectron_loader, []), (yt_faces_loader, []), (davis_loader, [])]

    # set up remaining training infrastructure
    devices = (args.devices * len(datasets))[:len(datasets)] # if there are more dataset than devices, distribute
    moments_loss = torch.nn.CrossEntropyLoss().cuda()
    objectron_loss = torch.nn.CrossEntropyLoss().cuda()
    yt_faces_loss = NT_Xent(0.1).cuda()
    davis_loss = torch.nn.BCEWithLogitsLoss().cuda()
    losses = [moments_loss, objectron_loss, yt_faces_loss, davis_loss]
    metrics = [(), (), (), ()]
    loggers = [Logger(args.logpath, args.ckptpath, logevery=args.ckptinterval)] * len(datasets)
    if args.resume:
        for i, l in enumerate(loggers):
            log = torch.load(logpath + f'rank{i}.log')
            l.losscurve = log['loss']
            l.metriccurve = log['metric']
        

    # launch training
    n = args.nprocs
    assert n == len(datasets), 'Number of training processes does not match number of datasets.'
    mp.spawn(multidata_train,
         args=(n, backbone, datasets, decoders, losses, metrics, devices, loggers, args.batches, args.loginterval, True),
         nprocs=n,
         join=True)
