'''
Wrapper / loader for Moments in Time dataset.
'''
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from data.utils import subsample

def read_csv(root_dir, phase):
    '''
    Reads the .csv files that contain all training set paths.
    
    Args:
        *root_dir: root directory of the dataset, which contains the csv files
        *phase (str): `training` or `validation`
    '''
    videos = []
    with open(os.path.join(root_dir, phase + 'Set.csv'), 'r') as f:
        for line in f:
            video, label, _, _ = line.split(',')
            videos.append((video, label))
    return(videos)

def get_categories(root_dir):
    '''
    Loads the mapping from category names to numeric labels from the file
    'moments_categories.txt' in the root_dir.
    '''
    category_map = {}
    with open(os.path.join(root_dir, 'moments_categories.txt'), 'r') as f:
        for line in f:
            category, label = line.split(',')
            label = int(label)
            category_map[category] = label
    return category_map


class MomentsDataset(Dataset):
    '''
    Wrapper for the Moments in Time dataset.
    
    Args:
        *root_dir (str): Directory from which to load data. Should contain files
                         `trainingSet.csv`, `validationSet.csv` and `moments_categories.txt`
                         as well as subfolders with training/validation data.
        *phase (str): `training` or `validation`
        *nframes (int): number of frames to subsample every video to
        *transform: PyTorch transform to apply to the videos/frames
    '''

    def __init__(self, root_dir, phase, nframes, transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.nframes = nframes
        self.transform = transform
        self.categories = get_categories(root_dir)
        self.videos = read_csv(root_dir, phase)
        self.videos = [(v[0], self.categories[v[1]]) for v in self.videos]
        # TODO: compute or load mean and standard deviation over complete dataset
        #       to normalize videos
        #       Alternatively, we can add a batch-norm layer at the front of the network
    
    def __len__(self):
        return(len(self.videos))
    
    # Getting one video takes about 90ms on my computer (without any transforms).
    # If that is not fast enough for us, we should think about precomputing the tensors
    # and saving them on disk.
    def __getitem__(self, idx):
        path, label = self.videos[idx]
        vid, _, _ = torchvision.io.read_video(os.path.join(self.root_dir, self.phase, path), 0.0, 3.0, 'sec')
        vid = subsample(vid, self.nframes)
        vid = vid.movedim(3, 0)
        if self.transform:
            vid = self.transform(vid)
        return(vid, label)

