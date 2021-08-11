'''
Wrapper / loader for Objectron dataset.

Assumes that data have been downloaded with `download_objectron_redux.py` script.
'''
import os
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import ConvertImageDtype, Resize, Compose, Lambda, Normalize
from data.utils import subsample

def get_paths(root_dir):
    '''
    Traverses the directory `root_dir` to get the paths of all objectron videos.
    '''
    classes = ['bike', 'book', 'bottle', 'camera', 'cereal_box', 'chair', 'cup', 'laptop', 'shoe']
    class2int = {c: i for (i, c) in enumerate(classes)}
    assert sorted(os.listdir(root_dir)) == classes, 'Directory contents do not match objectron classes.'
    paths = []
    for c in classes:
        cpaths = os.listdir(os.path.join(root_dir, c))
        paths.extend([(os.path.join(root_dir, c, p), class2int[c]) for p in cpaths])
    return paths


class ObjectronPreprocessor:
    '''
    Wrapper for the Objectron dataset.
    
    Args:
        *root_dir (str): Directory from which to load data. Should contain ...
        *nframes (int): number of frames to subsample every video to
        *transform: PyTorch transform to apply to the videos/frames
    '''

    def __init__(self, root_dir, nframes, transform=None):
        self.root_dir = root_dir
        self.nframes = nframes
        self.transform = transform
        self.videos = get_paths(root_dir)
    
    def __len__(self):
        return(len(self.videos))
    
    def __getitem__(self, idx):
        path, label = self.videos[idx]
        vid, _, _ = torchvision.io.read_video(path)
        
    def convert(self, idx):
        path, label = self.videos[idx]
        vid, _, _ = torchvision.io.read_video(path)
        vid = subsample(vid, self.nframes)
        vid = vid.movedim(3, 0)
        if self.transform:
            vid = self.transform(vid)
        basename = path.split('.')[0]
        torch.save(vid, basename + '.pt')
        

if __name__ == '__main__':
    transform = Compose([ConvertImageDtype(torch.float32), Resize((224, 224))])
    p = ObjectronPreprocessor('/data/objectron', 16, transform)
    
    print(f'Preprocessing {len(p)} videos. Time: {datetime.datetime.now()}', flush=True)
    for i in range(len(p)):
        p.convert(i)
        if i+1 % 100 == 0:
            print(f'Processed {i+1} videos. Time: {datetime.datetime.now()}', flush=True)
    
