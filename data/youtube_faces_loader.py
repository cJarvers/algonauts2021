'''
Wrapper / loader for YouTubeFaces dataset.
'''
import os
import math
import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import IterableDataset, DataLoader
import torchvision
from data.utils import subsample_ids

class YouTubeFacesFrameImagesDB():
    '''
    Underlying implementation for the `frame_images_DB` part of the
    YouTubeFaces dataset.

    Args:
        *root_dir (str): Directory from which to load data. Should contain a
                         folder `frame_images_DB` 
                         as well as subfolders with training/validation data.
        *split (str): `training` or `validation`
        *nframes (int): number of frames to subsample every video to
        *rng_state (int): initial state of the random number generator
    '''
    def __init__(self, root_dir, split, nsamples, rng_state=0):
        # keep track of which part of the split to be used later on
        self.split = split
        # how many frames to get from a video
        self.nsamples = nsamples
        self._init_rng(rng_state)
        # setup of metadata
        self._init_dataset_meta_info(root_dir)
        # split train/val
        self._perform_train_val_split()

    @classmethod
    def _get_subdirs(cls, dirpath):
        # cf. https://docs.python.org/3/library/pathlib.html#basic-use
        return [x for x in dirpath.iterdir() if x.is_dir()]

    @classmethod
    def _get_files_in_dir(cls, dirpath):
        return [x for x in dirpath.iterdir() if x.is_file()]

    @classmethod
    def _get_valid_persons(cls, person_dirs, person_infos):
        '''
        Reduces the list of potential persons in the dataset to only the valid
        candidates, i.e. for which at least one video and the label file exist.

        Args:
            *person_dirs: List of potential persons in the dataset
            *person_info: List of label files in the dataset
        '''
        valid_person_dirs = []
        for person_dir in person_dirs:
            if len(YouTubeFacesFrameImagesDB._get_subdirs(person_dir)) > 0 and \
               any(str(person_dir) in str(person_info) for person_info in person_infos):
               valid_person_dirs.append(person_dir)
        return valid_person_dirs


    @classmethod
    def _frames_to_video(cls, frame_paths):
        '''
        Reads images from a list of paths and concatenates them into a video.

        Args:
            *frame_paths: List of paths to single frames/images
        '''
        frames = []
        for frame_path in frame_paths:
            # yields Tensor[C, H, W]
            frames.append(torchvision.io.read_image(str(frame_path)))
        # put frames together
        # yields Tensor[T, C, H, W])
        vid = torch.stack(frames)
        # Rearrange as per our other datasets
        # yields Tensor[C, T, H, W]
        vid = vid.movedim(1, 0)
        return vid

    def _init_rng(self, rng_state):
        torch.manual_seed(rng_state)
        np.random.seed(rng_state)
        random.seed(rng_state)

    def _init_dataset_meta_info(self, root_dir):
        '''
        Parses lists of all persons of the dataset that make for valid samples,
        i.e. at least one video and the label file exist.

        Args:
            *root_dir: root directory of the dataset, which contains the data
        '''
        p = Path(root_dir) / "frame_images_DB"
        person_dirs = YouTubeFacesFrameImagesDB._get_subdirs(p)
        person_infos = list(p.glob('*.labeled_faces.txt'))

        valid_person_dirs = YouTubeFacesFrameImagesDB._get_valid_persons(person_dirs, person_infos)

        self.persons = np.sort(valid_person_dirs)
        self.labels = range(0,len(self.persons))

    def _perform_train_val_split(self):
        '''
        Perform a 80% train, 20% val split of the dataset.
        '''
        n_persons = len(self.persons)
        person_ids = np.random.permutation(range(n_persons))
        # split into 80% train, 20% val
        train_ids_end = np.ceil(0.8*n_persons).astype(int)
        self.train_ids = person_ids[:train_ids_end]
        self.validation_ids = person_ids[train_ids_end:]

    def get_ids(self):
        '''
        Returns the indices of the persons in the dataset belonging either to
        the training or validation split.
        '''
        if self.split == "training":
            return self.train_ids
        elif self.split == "validation":
            return self.validation_ids
        else:
            raise(ValueError("Encountered unknown split type, got: {}".format(self.split)))

    def _get_random_video(self, sample_dir):
        '''
        Returns a properly subsampled video stemming from a random subdirectory
        given the `sample_dir`.

        Args:
            *sample_dir: path of parent directory where videos are residing
        '''
        video_dirs = YouTubeFacesFrameImagesDB._get_subdirs(sample_dir)
        # choose from the available videos
        video_dir = np.random.choice(video_dirs)
        # select which frames to read from the video
        frames = YouTubeFacesFrameImagesDB._get_files_in_dir(video_dir)
        subsampled_frame_ids = subsample_ids(len(frames), self.nsamples)
        sampled_frames = [frames[idx] for idx in subsampled_frame_ids]

        return YouTubeFacesFrameImagesDB._frames_to_video(sampled_frames)

    def get_sample(self, sample_idx):
        '''
        Returns a random, properly subsampled video belonging to the person as
        indicated by the `sample_idx`.

        Args:
            *sample_idx (int): index indicating a person in the dataset
        '''
        sample_dir = self.persons[sample_idx]

        video = self._get_random_video(sample_dir)
        label = self.labels[sample_idx]

        return (video, label)

class YouTubeFacesDataset(IterableDataset):
    '''
    Wrapper for the YouTubeFaces. Should only be used with batch sizes that are
    multiples of `_max_sample_multiplicity`.

    Args:
        *root_dir (str): Directory from which to load data. Should contain a
                         folder `frame_images_DB` 
                         as well as subfolders with training/validation data.
        *phase (str): `training` or `validation`
        *nframes (int): number of frames to subsample every video to
        *transform: PyTorch transform to apply to the videos/frames
        *rng_state (int): initial state of the random number generator
    '''
    def __init__(self, root_dir, phase, nframes, transform, rng_state=0):
        self.root_dir = root_dir
        self.phase = phase
        self.nframes = nframes
        self.transform = transform
        self.dataset = YouTubeFacesFrameImagesDB(root_dir, phase, nframes, rng_state=rng_state)
        # which samples to process will usually be overwritten by
        # `worker_init_fn`
        self.samples_to_process = self.dataset.get_ids()
        self._iter_idx = 0
        # how often was the sample already passed to `__iter__`
        self._sample_multiplicity = 0
        # how often should the sample be passed to `__iter__`
        self._max_sample_multiplicity = 2
        # TODO: compute or load mean and standard deviation over complete dataset
        #       to normalize videos
        #       Alternatively, we can add a batch-norm layer at the front of the network

    def _get_paired_sample(self):
        '''
        Returns a sample of the underlying dataset. Ensures that as many as
        `_max_sample_multiplicity` consecutive calls to the method return
        samples from one and the same person (while potentially being
        different videos).
        '''
        # Only iterate further if not all samples have been retrieved
        if self._iter_idx >= len(self.samples_to_process):
            self._iter_idx = 0
            raise(StopIteration())

        vid, label = self.dataset.get_sample(self.samples_to_process[self._iter_idx])
        if self.transform:
            vid = self.transform(vid)
        # Handle internally tracked idx usages
        self._sample_multiplicity += 1
        if self._sample_multiplicity >= self._max_sample_multiplicity:
            self._sample_multiplicity = 0
            self._iter_idx += 1
        return (vid, label)

    def __iter__(self):
       return self

    def __next__(self):
       return self._get_paired_sample()

    @classmethod
    def worker_init_fn(cls, worker_id):
        '''
        Worker init function that configures each dataset copy differently as
        per the docs:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset

        Args:
            *worker_id (int): ID of the worker to perform the initalization for
        '''
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        # configure the dataset to only process the split workload
        sample_ids = dataset.dataset.get_ids()
        per_worker = int(math.ceil(len(sample_ids) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        worker_start = 0 + worker_id * per_worker
        worker_end = min(worker_start + per_worker, len(sample_ids))
        dataset.samples_to_process = sample_ids[worker_start:worker_end]
