'''Functions for multi-dataset training on cluster.'''
# Some parts of the distributed processing have been taken from the PyTorch tutorials:
# Source: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
#         Authors: Shen Li, Joe Zhu
#         BSD 3-Clause License
#         License Text:
#         Redistribution and use in source and binary forms, with or without
#         modification, are permitted provided that the following conditions are met:
#
#         * Redistributions of source code must retain the above copyright notice, this
#           list of conditions and the following disclaimer.
#
#         * Redistributions in binary form must reproduce the above copyright notice,
#           this list of conditions and the following disclaimer in the documentation
#           and/or other materials provided with the distribution.
#
#         * Neither the name of the copyright holder nor the names of its
#           contributors may be used to endorse or promote products derived from
#           this software without specific prior written permission.
#
#         THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#         AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#         IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#         DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#         FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#         DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#         SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#         CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#         OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#         OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# The bits that have been reused are marked.

# Standard Python imports
import datetime
from math import ceil
import os
import sys
import tempfile
# PyTorch-related imports
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# custom import from our code
from utils.utils import AverageMeter
from models.decoders import EncoderDecoderPair

# taken from PyTorch tutorial
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# taken from PyTorch tutorial
def cleanup():
    dist.destroy_process_group()


def trainstep(x, y, model, loss_fn, opt, dev):
    x = x.to(dev)
    y = y.to(dev)
    pred = model(x)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

def valloop(data, model, metric, dev, maxbatches=100):
    avgmetric = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(data):
            x = x.to(dev)
            y = y.to(dev)
            pred = model(x)
            m = metric(pred, y)
            avgmetric.update(loss.item())
            if i >= maxbatches:
                break
    model.train()
    return avgmetric.avg


def multidata_train(rank, world_size, make_backbone, datasets, decoders, losses, metrics,
        devices, loggers, batches=1000, loginterval=100, stepinterval=100, debug=False):
    '''
    Trains the common network `backbone` on several datasets simultaneously.

    Args:
        *rank (int): Process number that the current copy of the function is run on.
        *world_size (int): Number of processes on which the function is run in parallel.
        *make_backbone: Constructor for network component that is common across all datasets
        *datasets: The datasets to train on. Each should be a pair of iterables
                   (training and validation loaders).
        *decoders: The network components that differ for each dataset.
        *losses: Loss functions to use for training the network(s).
        *metrics: Functions to evaluate the network with on validation data.
        *devices: The compute devices on which the networks should be trained.
        *loggers: List of loggers (one per process). Should have a method .log that receives
                  the epoch number, batch number, average loss and metric,
                  model and decoder state_dicts, and process rank.
        *batches (int): Number of batches to train
        *loginterval (int): Number of batches after which to log loss and validation metric.
        *stepinterval (int): Number of batches after which to trigger the learning rate scheduler.
        *debug (bool): If True, prints some debug information

    The lists `datasets`, `decoders`, `losses`, and `devices` have to be of the same length.
    Essentially, device i will get a batch from dataset i, put it through the backbone,
    put the output of that through decoder i, calculate loss `i` and perform backprop.
    The gradients for the decoder are applied directly, the gradients for the backbone
    are accumulated across datasets / devices and applied together. 
    '''
    assert len(datasets) == len(decoders) == len(losses) == len(devices), \
        'Lists of datasets, decodes, losses, and devices have to have equal length.'
    # set up distributed processing and DDP model
    setup(rank, world_size)
    dev = devices[rank]
    model = make_backbone().to(dev)
    ddp_model = DDP(model, device_ids=[rank])
    decoder = decoders[rank].to(dev)
    complete_model = EncoderDecoderPair(ddp_model, decoder).to(dev)
    traindata, valdata = datasets[rank]

    # print some debug information
    if debug:
        print(f'Start on process {rank}: {datetime.datetime.now()}')
        print(f'Running multidata_train on process {rank}, device {dev}', flush=True)

    loss_fn = losses[rank]
    eval_fn = metrics[rank]
    optimizer = optim.SGD(complete_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # determine number of epochs according to number of batches
    epochs = ceil(batches / len(traindata))
    # set up logging infrastructure for training loop
    batchcounter = 0
    avgloss = AverageMeter()
    logger = loggers[rank]

    # run training loop
    for e in range(epochs):
        # train until interval for logging or checkpointing
        model.train()
        for i, (x, y) in enumerate(traindata):
            loss = trainstep(x, y, complete_model, loss_fn, optimizer, dev)
            avgloss.update(loss)
            batchcounter += 1
            if (batchcounter + 1) % stepinterval == 0:
                scheduler.step()
            if batchcounter % loginterval == 0:
                avgm = valloop(valdata, complete_model, eval_fn, dev)
                logger.log(e, batchcounter, avgloss.avg, avgm, model.state_dict(), decoder.state_dict(), rank)
                if debug:
                    print(f'Process {rank}, epoch {e}, batch {i+1}|{batchcounter}: loss {avgloss.avg} at time {datetime.datetime.now()}', flush=True)
                avgloss.reset()
            if batchcounter >= batches:
                break

    if debug:
        #print(f'Final network parameters on process {rank}: {list(model.parameters())}')
        #print(f'Final decoder parameters on process {rank}: {list(decoder.parameters())}')
        print(f'End on process {rank}: {datetime.datetime.now()}', flush=True)
    cleanup()

