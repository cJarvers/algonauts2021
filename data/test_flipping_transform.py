# quick and simple test script

print("Starting testing of Flipping transform")

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import datetime
import data.cityscapes_loader as csl
import data.davis_loader as dvl
import data.youtube_faces_loader as ytf
from data.utils import FlippingTransform
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ConvertImageDtype, Resize, Compose, Lambda, Normalize

def permutex(x):
    return(x.permute(1, 0, 2, 3))

def tolong(x):
    return(x.long())

def truncate(x):
    return(x.clamp(0, 1))

def squeeze(x):
    return(x.squeeze())

print("Imported module")

def test_flipping_transform_single_video():
    print("### test_flipping_transform_single_video start ###")
    ft = FlippingTransform(0.5)
    C,T,H,W = [3,16,244,244]
    vid = torch.rand((C,T,H,W))
    vid_orig = vid.detach().clone()
    vid_orig_flipped = torch.flip(vid_orig, [ft.flipping_axis])
    n_non_flipped = 0
    n_flipped = 0
    for _ in range(100):
        vid_ret = ft(vid.detach().clone())
        if torch.equal(vid_ret, vid_orig):
            n_non_flipped += 1
        elif torch.equal(vid_ret, vid_orig_flipped):
            n_flipped += 1
        else:
            raise ValueError("FlippingTransform not working as intended for single videos!")
    print("n_non_flipped: {}, n_flipped: {}".format(n_non_flipped, n_flipped))
    print("### test_flipping_transform_single_video finish ###")

def test_flipping_transform_video_list():
    print("### test_flipping_transform_video_list start ###")
    ft = FlippingTransform(0.5)
    C,T,H,W = [3,16,244,244]
    vid0 = torch.rand((C,T,H,W))
    vid0_orig = vid0.detach().clone()
    vid0_orig_flipped = torch.flip(vid0_orig, [ft.flipping_axis])
    vid1 = torch.rand((C,T,H,W))
    vid1_orig = vid1.detach().clone()
    vid1_orig_flipped = torch.flip(vid1_orig, [ft.flipping_axis])
    n_non_flipped = 0
    n_flipped = 0
    for _ in range(100):
        vids_ret = ft([vid0.detach().clone(),vid1.detach().clone()])
        if torch.equal(vids_ret[0], vid0_orig) and torch.equal(vids_ret[1], vid1_orig):
            n_non_flipped += 1
        elif torch.equal(vids_ret[0], vid0_orig_flipped) and torch.equal(vids_ret[1], vid1_orig_flipped):
            n_flipped += 1
        else:
            raise ValueError("FlippingTransform not working as intended for lists of videos!")
    print("n_non_flipped: {}, n_flipped: {}".format(n_non_flipped, n_flipped))
    print("### test_flipping_transform_video_list finish ###")

def test_yt_faces_dataloader():
    print("### YouTubeFaces dataloading start ###")
    print("Instantiating transform")
    transform = Compose([ConvertImageDtype(torch.float32), Resize((224, 224)), FlippingTransform(0.5)])
    print("Instantiated transform")
    print("Instantiating dataloader")
    dl = DataLoader(ytf.YouTubeFacesDataset('/data/YouTubeFaces','training', 16, transform),
           batch_size=32, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=8, collate_fn=None,
           pin_memory=False, drop_last=True, timeout=0,
           worker_init_fn=ytf.YouTubeFacesDataset.worker_init_fn, #prefetch_factor=2,
           persistent_workers=False)
    print("Instantiated dataloader")
    print("Iterating dataloader")
    it = 0
    print("Iteration: {} - {}".format(it, datetime.datetime.now()))
    for vl in dl:
        print("----")
        print("Iteration: {} - {}".format(it, datetime.datetime.now()))
        print("Shape: {}".format(vl[0].shape))
        print("Sample: {}".format(vl[1]))
        it += 1
        if it > 10:
            break
    print("---")
    print("Iterated dataloader")
    print("### YouTubeFaces dataloading finish ###")

def test_dv_dataloader():
    print("### DAVIS dataloading start ###")
    print("Instantiating transform")
    common_transform = Compose([FlippingTransform(0.5)])
    transform = Compose([ConvertImageDtype(torch.float32), Resize((224, 224))])
    label_transform = Compose([Lambda(squeeze), Lambda(tolong), Lambda(truncate), Resize((224, 224))])
    print("Instantiated transform")
    print("Instantiating dataloader")
    dl = DataLoader(dvl.DAVISDataset('/data/DAVIS','training', 16, transform, label_transform, common_transform),
           batch_size=4, shuffle=True, num_workers=4)
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
        if it > 10:
            break
    print("---")
    print("Iterated dataloader")
    print("### DAVIS dataloading finish ###")

def test_cs_dataloader():
    print("### Cityscapes dataloading start ###")
    print("Instantiating transform")
    common_transform = Compose([FlippingTransform(0.5)])
    obj_transform = Compose([Lambda(permutex), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        Lambda(permutex)])
    print("Instantiated transform")
    print("Instantiating dataloader")
    dl = DataLoader(csl.CityscapesDataset('/data/cityscapes', 'training', 16, obj_transform, None, common_transform, suffix='.pt'),
           batch_size=32, shuffle=True, num_workers=8)
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
        if it > 10:
            break
    print("---")
    print("Iterated dataloader")
    print("### Cityscapes dataloading finish ###")

if __name__ == '__main__':
    test_flipping_transform_single_video()
    test_flipping_transform_video_list()
    test_yt_faces_dataloader()
    test_dv_dataloader()
    test_cs_dataloader()
#    with torch.autograd.profiler.profile() as prof:
#        test_yt_faces_dataloader()
#    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
#    with torch.autograd.profiler.profile() as prof:
#        test_dv_dataloader()
#    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
