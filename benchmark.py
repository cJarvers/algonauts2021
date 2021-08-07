#!/usr/bin/env python3
'''Benchmark dataset loaders and training steps.'''

# standard Python imports
import argparse
import time
# PyTorch-related imports
import torch
import torch.multiprocessing as mp
from torchvision.transforms import ConvertImageDtype, Resize, Compose
from torch.utils.data import DataLoader
import torch.autograd.profiler as profiler
# our custom imports
from data.moments_loader import MomentsDataset
from data.objectron_loader import ObjectronDataset
from data.youtube_faces_loader import YouTubeFacesDataset
from data.davis_loader import DAVISDataset
from models.decoders import ClassDecoder
from models.resnet3d50 import ResNet3D50Backbone
from utils.training import multidata_train
from utils.utils import Logger
from utils.losses import NT_Xent

# set up command line parsing
parser = argparse.ArgumentParser(description='Perform benchmarking of data loaders and training.')
parser.add_argument('--bsize', type=int, default=32, help='Batch size')
parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Names of devices to train on.')
parser.add_argument('-n', '--num_workers', type=int, default=16, help='Number of workers per dataset loader.')

def timeit(fn, warmup=1, repeats=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t1 = time.perf_counter()
        fn()
        t2 = time.perf_counter()
        times.append(t2 - t1)
    return(min(times), sum(times) / len(times), max(times))

if __name__ == '__main__':
    # parse command line arguments and check that environment is set up
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'Script requires GPU, but cuda not available.'
    dev = args.device

    # set up model and decoders
    backbone = ResNet3D50Backbone().to(dev)
    moments_decoder = ClassDecoder(305).to(dev)
    objectron_decoder = ClassDecoder(9).to(dev)
    youtube_faces_decoder = ClassDecoder(64).to(dev)

    # load datasets
    n = args.num_workers
    transform = Compose([ConvertImageDtype(torch.float32), Resize((224, 224))])
    moments = MomentsDataset('/data/Moments_in_Time_Raw', 'training', 16, transform=transform)
    moments_loader = DataLoader(moments, batch_size=args.bsize, shuffle=True, num_workers=n)
    objectron = ObjectronDataset('/data/objectron', 16, transform=transform)
    objectron_loader = DataLoader(objectron, batch_size=args.bsize, shuffle=True, num_workers=n)
    yt_faces = YouTubeFacesDataset('/data/YouTubeFaces', 'training', 16, transform=transform)
    # shuffling for yt_faces happens within the dataset implementation
    # as it's an iterable dataset
    yt_faces_loader = DataLoader(yt_faces, batch_size=args.bsize, shuffle=False, drop_last=True,
                                 worker_init_fn=YouTubeFacesDataset.worker_init_fn, num_workers=n)
    davis = DAVISDataset('/data/DAVIS', 'training', 16, common_transform=transform, augmentation_transform=lambda x: x)
    davis_loader = DataLoader(davis, batch_size=args.bsize, shuffle=True, num_workers=n)

    # set up remaining training infrastructure
    moments_loss = torch.nn.CrossEntropyLoss().cuda()
    objectron_loss = torch.nn.CrossEntropyLoss().cuda()
    yt_faces_loss = NT_Xent(0.1).cuda()

    ##### Profile data loaders #####
    # Moments in Time
    print('Profiling Moments in Time data loading')
    iterator = iter(moments_loader)
    t_min, t_avg, t_max = timeit(lambda: next(iterator))
    print(f'Time to load one batch: {t_min} - {t_avg} - {t_max}')
    #with profiler.profile(profile_memory=True) as prof:
    #    x, y = next(iterator)
    #print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print()

    # objectron
    print('Profiling objectron data loading')
    iterator = iter(objectron_loader)
    t_min, t_avg, t_max = timeit(lambda: next(iterator))
    print(f'Time to load one batch: {t_min} - {t_avg} - {t_max}')
    #with profiler.profile(profile_memory=True) as prof:
    #    x, y = next(iterator)
    #print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print()

    # youtube faces
    print('Profiling Youtube Faces data loading')
    iterator = iter(yt_faces_loader)
    t_min, t_avg, t_max = timeit(lambda: next(iterator))
    print(f'Time to load one batch: {t_min} - {t_avg} - {t_max}')
    #with profiler.profile(profile_memory=True) as prof:
    #    x, y = next(iterator)
    #print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print()

    # DAVIS
    print('Profiling DAVIS data loading')
    iterator = iter(davis_loader)
    t_min, t_avg, t_max = timeit(lambda: next(iterator), repeats=2)
    print(f'Time to load one batch: {t_min} - {t_avg} - {t_max}')
    #with profiler.profile(profile_memory=True) as prof:
    #    x, y = next(iterator)
    #print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print()

    ##### Profile feedforward passes
    print('Profiling moments in time feedforward pass:')
    iterator = iter(moments_loader)
    x, y = next(iterator)
    x = x.to(dev)
    y = y.to(dev)
    t_min, t_avg, t_max = timeit(lambda: moments_decoder(backbone(x)))
    print(f'Time for one feedforward pass: {t_min} - {t_avg} - {t_max}')
    #with profiler.profile(use_cuda=True, profile_memory=True) as prof:
    #    pred = moments_decoder(backbone(x))
    #print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))
    print()
    del x, y, pred


    ##### Profile backward computation
    print('Profiling Moments in Time backward pass:')
    iterator = iter(moments_loader)
    x, y = next(iterator)
    x = x.to(dev)
    y = y.to(dev)
    pred = moments_decoder(backbone(x))
    with profiler.profile(use_cuda=True, profile_memory=True) as prof:
        loss = moments_loss(pred, y)
        loss.backward()
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))
    print()
