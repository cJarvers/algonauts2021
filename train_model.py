#!/usr/bin/env python3
'''Perform multiGPU training of 3D-ResNet50 model on multiple datasets.'''
# TODO:
# - Currently, frames are only converted to float, resized to 224x224 and normalized.
#   We may want to add augmentations.
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
from torchvision.transforms import ConvertImageDtype, Resize, Compose, Lambda, Normalize, RandomErasing, GaussianBlur, RandomGrayscale, RandomApply, ColorJitter
from torch.utils.data import DataLoader
# our custom imports
from data.moments_loader import MomentsDataset
from data.objectron_loader import ObjectronDataset
from data.youtube_faces_loader import YouTubeFacesDataset
from data.davis_loader import DAVISDataset
from data.cityscapes_loader import CityscapesDataset
from data.utils import FlippingTransform
from models.decoders import ClassDecoder, UNet3DDecoder, Deconv2DDecoder
from models.resnet3d50 import ResNet3D50Backbone
from utils.training import multidata_train
from utils.utils import Logger
from utils.losses import NT_Xent

# set up command line parsing
parser = argparse.ArgumentParser(description='Perform training of 3D-ResNet50 model on multiple datasets.')
parser.add_argument('--bsize', type=int, default=32, help='Batch size')
parser.add_argument('-d', '--devices', type=str, nargs='*', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4'], help='Names of devices to train on.')
parser.add_argument('-n', '--nprocs', type=int, default=5, help='Number of processes to launch.')
parser.add_argument('-b', '--batches', type=int, default=1000, help='Number of batches to train.')
parser.add_argument('--logpath', type=str, default='/mnt/logs/', help='Path to save log files to.')
parser.add_argument('--ckptpath', type=str, default='/mnt/logs/', help='Path to save checkpoints to.')
parser.add_argument('--loginterval', type=int, default=1000, help='Number of batches after which to perform validation and log losses & metrics.')
parser.add_argument('--stepinterval', type=int, default=1000, help='Number of batches after which to step the learning rate scheduler.')
parser.add_argument('--ckptinterval', type=int, default=1000, help='Number of batches after which to save logs and checkpoints (should be a multiple of loginterval).')
parser.add_argument('--resume', dest='resume', action='store_true', help='Resume training from pervious checkpoint.')
parser.set_defaults(resume=False)

def permutex(x):
    return(x.permute(1, 0, 2, 3))

def tolong(x):
    return(x.long())

def truncate(x):
    return(x.clamp(0, 1))

def squeeze(x):
    return(x.squeeze())

if __name__ == '__main__':
    # parse command line arguments and check that environment is set up
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'Script requires GPU, but cuda not available.'

    # set up model and decoders
    backbone = ResNet3D50Backbone
    moments_decoder = ClassDecoder(305)
    objectron_decoder = ClassDecoder(9)
    youtube_faces_decoder = ClassDecoder(64)
    davis_decoder = UNet3DDecoder(inplanes=[2048, 2048, 1024, 512, 128],
        planes=[512, 256, 128, 64, 64], outplanes=[1024, 512, 256, 64, 64],
        upsample=[True, True, True, False, True],
        finallayer=torch.nn.ConvTranspose3d(64, 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)))
    cityscapes_decoder = Deconv2DDecoder(inplanes=2048, planes=[512, 256, 128, 64, 64],
        outplanes=[1024, 512, 256, 128, 64], upsample=[True, True, True, True, True],
        finallayer=torch.nn.Conv2d(64, 34, kernel_size=1))
    # set up for resuming job
    batchnum = 0
    if args.resume: # to resume previous training, load weights from previous checkpoint
        log = torch.load(args.logpath + 'rank0.log')
        batchnum = log['loss'][-1][1]
        # load weights
        weights = torch.load(args.ckptpath + f'model_0_b{batchnum}.ckpt')
        backbone.load_state_dict(weights)
        weights = torch.load(args.ckptpath + f'decoder_0_b{batchnum}.ckpt')
        moments_decoder.load_state_dict(weights)
        weights = torch.load(args.ckptpath + f'decoder_1_b{batchnum}.ckpt')
        objectron_decoder.load_state_dict(weights)
        weights = torch.load(args.ckptpath + f'decoder_2_b{batchnum}.ckpt')
        youtube_faces_decoder.load_state_dict(weights)
        weights = torch.load(args.ckptpath + f'decoder_3_b{batchnum}.ckpt')
        davis_decoder.load_state_dict(weights)
        weights = torch.load(args.ckptpath + f'decoder_4_b{batchnum}.ckpt')
        cityscapes.load_state_dict(weights)
    decoders = [moments_decoder, objectron_decoder, youtube_faces_decoder, davis_decoder, cityscapes_decoder]

    # set up transforms to apply to data
    fromfile = Compose([ConvertImageDtype(torch.float32), Resize((224, 224))])
    reshape = Lambda(permutex)
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    augment = Compose([RandomGrayscale(p=0.2), RandomApply([GaussianBlur((7,7))], p=0.2), RandomApply([ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.2), RandomErasing(p=0.2)])
    moments_transform = Compose([fromfile, reshape, augment, normalize, reshape])
    objectron_transform = Compose([reshape, augment, normalize, reshape])
    yt_transform = Compose([fromfile, reshape, augment, normalize, reshape])
    davis_transform = Compose([fromfile, reshape, augment, normalize, reshape])
    label_transform = Compose([Lambda(squeeze), Lambda(tolong), Lambda(truncate), Resize((224, 224))])
    common_flip = Compose([FlippingTransform(0.2)])
    cityscapes_transform = Compose([reshape, augment, normalize, reshape])
    # load datasets
    moments = MomentsDataset('/data/Moments_in_Time_Raw', 'training', 16, transform=moments_transform)
    moments_loader = DataLoader(moments, batch_size=args.bsize, shuffle=True, drop_last=True, num_workers=args.bsize)
    objectron = ObjectronDataset('/data/objectron', 16, transform=objectron_transform, suffix='.pt')
    objectron_loader = DataLoader(objectron, batch_size=args.bsize, shuffle=True, drop_last=True, num_workers=args.bsize)
    yt_faces = YouTubeFacesDataset('/data/YouTubeFaces', 'training', 16, transform=yt_transform)
    # shuffling for yt_faces happens within the dataset implementation
    # as it's an iterable dataset
    yt_faces_loader = DataLoader(yt_faces, batch_size=args.bsize, shuffle=False, drop_last=True,
                                 worker_init_fn=YouTubeFacesDataset.worker_init_fn, num_workers=args.bsize)
    davis = DAVISDataset('/data/DAVIS', 'training', 16, davis_transform, label_transform, common_flip)
    davis_loader = DataLoader(davis, batch_size=4, shuffle=True, drop_last=True, num_workers=4)
    cityscapes = CityscapesDataset('/data/cityscapes', 'training', 16, cityscapes_transform, None, common_flip, suffix='.pt')
    cityscapes_loader = DataLoader(cityscapes, batch_size=args.bsize, shuffle=True, drop_last=True, num_workers=args.bsize)
    datasets = [(moments_loader, []), (objectron_loader, []), (yt_faces_loader, []), (davis_loader, []), (cityscapes_loader, [])]

    # set up remaining training infrastructure
    devices = (args.devices * len(datasets))[:len(datasets)] # if there are more dataset than devices, distribute
    moments_loss = torch.nn.CrossEntropyLoss().cuda()
    objectron_loss = torch.nn.CrossEntropyLoss().cuda()
    yt_faces_loss = NT_Xent(0.1).cuda()
    davis_loss = torch.nn.CrossEntropyLoss().cuda()
    cityscapes_loss = torch.nn.CrossEntropyLoss().cuda()
    losses = [moments_loss, objectron_loss, yt_faces_loss, davis_loss, cityscapes_loss]
    metrics = [(), (), (), (), ()]
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
         args=(n, backbone, datasets, decoders, losses, metrics, devices, loggers, batchnum, args.batches, args.loginterval, args.stepinterval, True),
         nprocs=n,
         join=True)
