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
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# taken from PyTorch tutorial
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# taken from PyTorch tutorial
def cleanup():
    dist.destroy_process_group()


def multidata_train(rank, world_size, make_backbone, datasets, decoders, losses, devices):
    '''
    Trains the common network `backbone` on several datasets simultaneously.
    
    Args:
        *rank (int): Process number that the current copy of the function is run on.
        *world_size (int): Number of processes on which the function is run in parallel.
        *make_backbone: Constructor for network component that is common across all datasets
        *datasets: The datasets to train on. Each should be an iterable that yields batches.
        *decoders: The network components that differ for each dataset.
        *devices: The compute devices on which the networks should be trained.
        
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
    complete_model = nn.Sequential(ddp_model, decoder)
    
    loss_fn = losses[rank]
    optimizer = optim.SGD(complete_model.parameters(), lr=0.001)
    
    # run training loop
    for (x, y) in datasets[rank]:
        optimizer.zero_grad()
        x = x.to(dev)
        y = y.to(dev)
        y_hat = complete_model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
    
    cleanup()
    
    
# taken from PyTorch tutorial
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_2losses():
    decoder1 = nn.Linear(5, 2)
    decoder2 = nn.Linear(5, 3)
    xs1 = [torch.rand(1, 10), torch.rand(5, 10), torch.rand(3, 10)]
    xs2 = [torch.rand(2, 10), torch.rand(2, 10), torch.rand(2, 10)]
    ys1 = [torch.rand(1, 2), torch.rand(5, 2), torch.rand(3, 2)]
    ys2 = [torch.randint(3, (2,))] * 3
    data1 = zip(xs1, ys1)
    data2 = zip(xs2, ys2)
    loss1 = torch.nn.MSELoss()
    loss2 = torch.nn.CrossEntropyLoss()
    dev1 = torch.device('cuda:0')
    dev2 = torch.device('cuda:1')
    # adjusted from PyTorch tutorial
    mp.spawn(multidata_train,
             args=(2, ToyModel, [data1, data2], [decoder1, decoder2], [loss1, loss2], [dev1, dev2]),
             nprocs=2,
             join=True)

