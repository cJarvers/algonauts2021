'''
Wrapper / loader for Objectron dataset.

Assumes that data have been downloaded with `download_objectron_redux.py` script.
'''
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from data.utils import subsample

def get_paths(root_dir, suffix):
    '''
    Traverses the directory `root_dir` to get the paths of all objectron videos.
    '''
    classes = ['bike', 'book', 'bottle', 'camera', 'cereal_box', 'chair', 'cup', 'laptop', 'shoe']
    class2int = {c: i for (i, c) in enumerate(classes)}
    assert sorted(os.listdir(root_dir)) == classes, 'Directory contents do not match objectron classes.'
    paths = []
    for c in classes:
        cpaths = os.listdir(os.path.join(root_dir, c))
        paths.extend([(os.path.join(root_dir, c, p), class2int[c]) for p in cpaths if p[-len(suffix):] == suffix])
    return paths


class ObjectronDataset(Dataset):
    '''
    Wrapper for the Objectron dataset.
    
    Args:
        *root_dir (str): Directory from which to load data. Should contain ...
        *nframes (int): number of frames to subsample every video to
        *transform: PyTorch transform to apply to the videos/frames
        *suffix (str): file suffix to load data from; .pt or .mov
    '''

    def __init__(self, root_dir, nframes, transform=None, suffix='.pt'):
        self.root_dir = root_dir
        self.nframes = nframes
        self.transform = transform
        self.videos = get_paths(root_dir, suffix)
        # TODO: compute or load mean and standard deviation over complete dataset
        #       to normalize videos
        #       Alternatively, we can add a batch-norm layer at the front of the network
    
    def __len__(self):
        return(len(self.videos))
    
    # Getting one video takes about ... on my computer (without any transforms).
    # If that is not fast enough for us, we should think about precomputing the tensors
    # and saving them on disk.
    def __getitem__(self, idx):
        path, label = self.videos[idx]
        if self.suffix == '.pt':
            vid = torch.load(path)
        else:
            vid, _, _ = torchvision.io.read_video(path)
            vid = subsample(vid, self.nframes)
            vid = vid.movedim(3, 0)
        if self.transform:
            vid = self.transform(vid)
        return(vid, label)

