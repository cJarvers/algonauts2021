# quick and simple test script

print("Starting testing of YouTubeFaces dataloader")

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import datetime
import data.davis_loader as dvl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ConvertImageDtype, Resize, Compose

print("Imported module")

def test_db():
    print("Instantiating DB")
    db = dvl.DAVISDB('/data/DAVIS','training',16)
    print("Instantiated DB")
    print("Getting IDs")
    print(db.get_ids())
    print("Gotten IDs")
    print("Getting sample")
    v,l = db.get_sample(db.get_ids()[0])
    print(v)
    print(l)
    print(v.shape)
    print(l.shape)
    print("Gotten sample")

def test_dataset():
    print("Instantiating dataset")
    ds = dvl.DAVISDataset('/data/DAVIS','training',16, None, None)
    print("Instantiated dataset")
    print("Iterating dataset")
    for (idx, vl) in enumerate(ds):
        print("----")
        print("Iteration: {}".format(idx))
        print("Shape: {}".format(vl[0].shape))
        print("Sample: {}".format(vl[1].shape))
        if idx >= 50:
            print(vl[0])
            print(vl[1])
            break
    print("---")
    print("Iterated dataset")

def test_dataloader():
    print("Instantiating transform")
    transform = Compose([ConvertImageDtype(torch.float32), Resize((224, 224))])
    print("Instantiated transform")
    print("Instantiating dataloader")
    dl = DataLoader(dvl.DAVISDataset('/data/DAVIS','training',16, transform, None),
           batch_size=32, shuffle=True, num_workers=2)
    print("Instantiated dataloader")
    print("Iterating dataloader")
    it = 0
    print("Iteration: {} - {}".format(it, datetime.datetime.now()))
    for vl in dl:
        print("----")
        print("Iteration: {} - {}".format(it, datetime.datetime.now()))
        print("Shape: {}".format(vl[0].shape))
        print("Sample: {}".format(vl[1].shape))
        it += 1
    print("---")
    print("Iterated dataloader")

if __name__ == '__main__':
#    test_db()
#    test_dataset()
#    test_dataloader()
    with torch.autograd.profiler.profile() as prof:
#        test_dataset()
        test_dataloader()
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
