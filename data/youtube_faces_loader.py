'''
Wrapper / loader for YouTubeFaces dataset.
'''
import os
import math
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import IterableDataset, DataLoader
import torchvision
from utils import subsample_ids

class YouTubeFacesFrameImagesDB():
    def __init__(self, root_dir, split, nsamples, rng_state=0):
        # TODO: set initial rng state
        self._init_rng(rng_state)
        # setup of metadata
        self._init_dataset_meta_info(root_dir)
        # TODO: how to setup train/val split for this dataset?
        # split train/val
        self._perform_train_val_split()
        # keep track of which part of the split to be used later on
        self.split = split
        self.nsamples = nsamples

    @classmethod
    def _get_subdirs(cls, dirpath):
        # cf. https://docs.python.org/3/library/pathlib.html#basic-use
        return [x for x in dirpath.iterdir() if x.is_dir()]

    @classmethod
    def _get_files_in_dir(cls, dirpath):
        return [x for x in dirpath.iterdir() if x.is_file()]

    @classmethod
    def _get_valid_persons(cls, person_dirs, person_infos):
        valid_person_dirs = []
        for person_dir in person_dirs:
            if len(YouTubeFacesFrameImagesDB._get_subdirs(person_dir)) > 0 and \
               any(str(person_dir) in str(person_info) for person_info in person_infos):
               valid_person_dirs.append(person_dir)
        return valid_person_dirs


    @classmethod
    def _frames_to_video(cls, frame_paths):        
        frames = []
        for frame_path in frame_paths:
            frames.append(torchvision.io.read_image(frame_path))
        # put frames together the same way as `torchvision` would do:
        # Tensor[T, H, W, C]): the `T` video frames
        vid = torch.stack(frames)
        # Rearrange as per our other datasets
        vid = vid.movedim(3, 0)
        return vid
        
    def _init_rng(self, rng_state):
        pass

    def _init_dataset_meta_info(self, root_dir):
        '''
        Reads the .csv files that contain all training set paths.
        
        Args:
            *root_dir: root directory of the dataset, which contains the csv files
            *phase (str): `training` or `validation`
        '''
        p = Path(root_dir) / "frame_images_DB"
        person_dirs = YouTubeFacesFrameImagesDB._get_subdirs(p)
        person_infos = list(p.glob('*.labeled_faces.txt'))

        valid_person_dirs = YouTubeFacesFrameImagesDB._get_valid_persons(person_dirs, person_infos)

        self.persons = np.sort(valid_person_dirs)
        self.labels = range(0,len(self.persons))

    def _perform_train_val_split(self):
        n_persons = len(self.persons)
        person_ids = np.random.permute(range(n_persons))
        # split into 80% train, 20% val
        train_ids_end = np.ceil(0.8*n_persons).astype(int)
        self.train_ids = person_ids[:train_ids_end]
        self.validation_ids = person_ids[train_ids_end:]

    def get_ids(self):
        if self.split == "train":
            return self.train_ids
        elif self.split == "validation":
            return self.validation_ids
        else:
            raise(ValueError("Encountered unknown split type, got: {}".format(self.split)))

    def _get_random_sample_idx(self):
        return np.random.choice(self.get_ids())

    def _get_random_video(self, sample_dir):        
        video_dirs = YouTubeFacesFrameImagesDB._get_subdirs(sample_dir)
        # choose from the available videos
        video_dir = np.random.choice(video_dirs)
        # select which frames to read from the video
        frames = YouTubeFacesFrameImagesDB._get_files_in_dir(video_dir)        
        subsampled_frame_ids = subsample_ids(len(frames), self.nsamples)

        return YouTubeFacesFrameImagesDB._frames_to_video(frames[subsampled_frame_ids])

    def get_sample(self, sample_idx):
        sample_dir = self.persons[sample_idx]

        video = self._get_random_video(sample_dir)
        label = self.labels[sample_idx]

        return (video, label)

class YouTubeFacesDataset(IterableDataset):
    '''
    Wrapper for the YouTubeFaces. Currently only the `frame_images_DB` part of
    the dataset is supported.
    
    Args:
        *root_dir (str): Directory from which to load data. Should contain a
                         folder `frame_images_DB` 
                         as well as subfolders with training/validation data.
        *phase (str): `training` or `validation`
        *nframes (int): number of frames to subsample every video to
        *transform: PyTorch transform to apply to the videos/frames
    '''
    def __init__(self, root_dir, phase, nframes, transform):
        #TODO
        self.root_dir = root_dir
        self.phase = phase
        self.nframes = nframes
        self.dataset = YouTubeFacesFrameImagesDB(root_dir, phase, nframes)
        # will be usually overwritten by `worker_init_fn`
        self.samples_to_process = self.dataset.get_ids()
        self._iter_idx = 0
        # how often was the sample already passed to `__iter__`
        self._sample_multiplicity = 0
        # how often should the sample be passed to `__iter__`
        self._max_sample_multiplicity = 0
        # TODO: compute or load mean and standard deviation over complete dataset
        #       to normalize videos
        #       Alternatively, we can add a batch-norm layer at the front of the network

    def __iter__(self):
        vid, label = self.dataset.get_sample(self._iter_idx)
        if self.transform:
            vid = self.transform(vid)
        # Handle internally tracked idx usages
        self._sample_multiplicity += 1
        if self._sample_multiplicity >= self._max_sample_multiplicity:
            self._sample_multiplicity = 0
            self._iter_idx += 1
            if self._iter_idx >= len(self.samples_to_process):
                self._iter_idx = 0
        return(vid, label)

# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    # configure the dataset to only process the split workload
    sample_ids = dataset.dataset.get_ids()
    per_worker = int(math.ceil(len(sample_ids) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    worker_start = 0 + worker_id * per_worker
    worker_end = min(dataset.start + per_worker, len(sample_ids))
    dataset.samples_to_process = sample_ids[worker_start:worker_end]