'''
Wrapper / loader for Cityscapes dataset.
'''
import os
import math
from typing import Dict
import numpy as np
import random
import pathlib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from data.utils import subsample_ids

class CityscapesDB():
    '''
    Underlying implementation for the Cityscapes dataset.

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
        self.root_dir = root_dir
        self._init_dataset_meta_info(root_dir)
        # split train/val
        self._perform_train_val_split(root_dir)

    @classmethod
    def _get_subdirs(cls, dirpath):
        # cf. https://docs.python.org/3/library/pathlib.html#basic-use
        return [x for x in dirpath.iterdir() if x.is_dir()]

    @classmethod
    def _get_files_in_dir(cls, dirpath):
        return [x for x in dirpath.iterdir() if x.is_file()]

    @classmethod
    def _get_valid_videos(cls, vid_dir, ann_dir):
        '''
        Reduces the list of potential videos in the dataset to only the valid
        candidates, i.e. for which at least one video and the video exist.

        Args:
            *vid_dir: List of potential videos in the dataset
            *ann_dir: List of annotation videos in the dataset
        '''
        sequences = CityscapesDB._get_files_in_dir(vid_dir)
        # get the unique sequences
        seq_strs = [str(frame.stem).split("_")[1] for frame in sequences if not str(frame)[-3:] == '.pt']
        seqs = np.unique(seq_strs)

        # there should be an annotation per sequence
        valid_seqs = []
        for seq in seqs:
            city_name = str(ann_dir).split(pathlib.os.sep)[-1]
            label = list(ann_dir.glob(city_name + "_" + seq + "*_labelIds.png"))
            if len(label) == 0:
                continue
            frames = list(vid_dir.glob(city_name + "_" + seq + "*.png"))
            if len(frames) != 30:
                continue
            valid_seqs.append(seq)

        return valid_seqs

    @classmethod
    def _read_image(cls, path_to_img):
        # yields Tensor[H, W, C]
        return torch.from_numpy(np.array(Image.open(str(path_to_img))))


    @classmethod
    def _frames_to_video(cls, frame_paths):
        '''
        Reads images from a list of paths and concatenates them into a video.

        Args:
            *frame_paths: List of paths to single frames/images
        '''
        frame_0 = CityscapesDB._read_image(frame_paths[0])
        vid = torch.zeros((len(frame_paths),*frame_0.shape),dtype=frame_0.dtype) # Tensor [T, H, W, C]
        vid[0,:] = frame_0
        for (frame_idx, frame_path) in enumerate(frame_paths[1:]):
            vid[frame_idx,:] = CityscapesDB._read_image(frame_path)
        # yields Tensor[C, T, H, W]
        if len(vid.shape) == 4:
            vid = vid.movedim(3, 0)
        else:
            vid = torch.unsqueeze(vid,0)
        return vid

    @classmethod
    def _subsample_specific_ids(cls, nframes, nsamples, anchor_id):
        # subdivide by the proper ratio into ids previous and after the
        # `anchor_id`
        ratio_post_pre = (nframes - anchor_id) / nframes
        nsamples_pre = np.ceil((nsamples-1) * (1.0-ratio_post_pre)).astype(int)
        nsamples_post = np.floor((nsamples-1) * ratio_post_pre).astype(int)

        ids_prev = subsample_ids(anchor_id, nsamples_pre)
        ids_post = np.array(subsample_ids(nframes - anchor_id, nsamples_post)) + anchor_id

        return [*ids_prev, anchor_id, *ids_post]

    def _init_rng(self, rng_state):
        torch.manual_seed(rng_state)
        np.random.seed(rng_state)
        random.seed(rng_state)

    def _init_dataset_meta_info(self, root_dir):
        '''
        Parses lists of all videos of the dataset that make for valid samples,
        i.e. at least one video and the annotation videos exist.

        Args:
            *root_dir: root directory of the dataset, which contains the data
        '''

        # Dataset structure is {root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}
        # according to https://github.com/mcordts/cityscapesScripts
        vids = Path(root_dir) / "leftImg8bit_sequence_trainvaltest" / "leftImg8bit_sequence"
        anns = Path(root_dir) / "gtFine_trainvaltest" / "gtFine"
        phase_dirs = CityscapesDB._get_subdirs(vids)

        valid_vids = dict()

        for phase_dir in phase_dirs:
            city_dirs = CityscapesDB._get_subdirs(phase_dir)
            phase_dir_name = str(phase_dir).split(pathlib.os.sep)[-1]
            for city_dir in city_dirs:
                city_dir_name = str(city_dir).split(pathlib.os.sep)[-1]
                vids = CityscapesDB._get_valid_videos(city_dir, anns / phase_dir_name / city_dir_name)
                if len(vids) == 0:
                    continue
                if not phase_dir_name in valid_vids.keys():
                    valid_vids[phase_dir_name] = dict()
                valid_vids[phase_dir_name][city_dir_name] = vids

        self.videos = valid_vids

    def _perform_train_val_split(self, root_dir):
        '''
        Make sure relevant splits are available
        '''

        if not "train" in self.videos.keys() or not "val" in self.videos.keys():
            raise KeyError("Missing split in dataset")

    def get_ids(self):
        '''
        Returns the indices of the persons in the dataset belonging either to
        the training or validation split.
        '''
        if self.split == "training":
            city_keys_sorted = np.sort([*self.videos["train"].keys()])
            ids = []
            for city_key in city_keys_sorted:
                seqs = np.sort(self.videos["train"][city_key])
                ids.append(np.stack([[city_key] * len(seqs), seqs],axis=1))
            return np.concatenate(ids)
        elif self.split == "validation":
            city_keys_sorted = np.sort([*self.videos["val"].keys()])
            ids = []
            for city_key in city_keys_sorted:
                seqs = np.sort(self.videos["val"][city_key])
                ids.append(np.stack([[city_key] * len(seqs), seqs],axis=1))
            return np.concatenate(ids)
        else:
            raise(ValueError("Encountered unknown split type, got: {}".format(self.split)))

    def _get_video(self, split_name, city_name, seq_name):
        '''
        Returns a properly subsampled video stemming from the given `video_dir`.

        Args:
            *video_dir: path to the directory where images are residing
        '''
        if not seq_name in self.videos[split_name][city_name]:
            raise KeyError("Requested sequence not in valid videos")
        vids = Path(self.root_dir) / "leftImg8bit_sequence_trainvaltest" / "leftImg8bit_sequence"
        anns = Path(self.root_dir) / "gtFine_trainvaltest" / "gtFine"
        # select which frames to read from the video
        frames = np.sort([str(frame) for frame in (vids / split_name / city_name).glob(city_name + "_" + seq_name + "*.png")])
        anns = np.sort([str(frame) for frame in (anns / split_name / city_name).glob(city_name + "_" + seq_name + "*_labelIds.png")])
        if not len(anns) == 1:
            raise ValueError("Unexpected amount of annotations")

        subsampled_frame_ids = CityscapesDB._subsample_specific_ids(len(frames), self.nsamples, int(Path(anns[0]).stem.split("_")[2]))
        sampled_frames = [frames[idx] for idx in subsampled_frame_ids]

        return CityscapesDB._frames_to_video(sampled_frames), CityscapesDB._frames_to_video(anns)

    def get_sample(self, sample_idx):
        '''
        Returns a random, properly subsampled video belonging to the person as
        indicated by the `sample_idx`.

        Args:
            *sample_idx (int): index indicating a person in the dataset
        '''
        #map idx to keys
        (city_key, seq_key) = self.get_ids()[sample_idx]
        if self.split == "training":
            video, annotation = self._get_video("train", city_key, seq_key)
        elif self.split == "validation":
            video, annotation = self._get_video("val", city_key, seq_key)
        else:
            raise ValueError("Encountered unknown split")

        return (video, annotation)

class CityscapesDataset(Dataset):
    '''
    Wrapper for the Cityscapes dataset.
    Args:
        *root_dir (str): Directory from which to load data. Should contain
                         subfolders `Annotations`, `ImageSets`, and `JPEGImages`.
        *phase (str): `training` or `validation`
        *nframes (int): number of frames to subsample every video to
        *transform: PyTorch transform to apply to the video frames
        *label_transform: PyTorch transform to apply to the label frames
        *common_transform: PyTorch transform to apply to both the images and the labels
        *suffix (str): file suffix to load data from; .pt or .png
        *rng_state (int): initial state of the random number generator
    '''
    def __init__(self, root_dir, phase, nframes, transform=None, label_transform=None, common_transform=None, suffix='.pt', rng_state=0):
        self.root_dir = root_dir
        self.phase = phase
        self.nframes = nframes
        self.transform = transform
        self.label_transform = label_transform
        self.common_transform = common_transform
        self.dataset = CityscapesDB(root_dir, phase, nframes, rng_state=rng_state)
        self.suffix = suffix

    def _shortphase(self):
        if self.phase == 'training':
            return('train')
        elif self.phase == 'validation':
            return('val')
        elif self.phase == 'test':
            return('test')
        else:
            raise ValueError(f'Encountered unknown phase: {self.phase}')

    def __getitem__(self, idx):
        if self.suffix == '.png':
            (vid, ann) = self.dataset.get_sample(idx)
        elif self.suffix == '.pt':
            (city_key, seq_key) = self.dataset.get_ids()[idx]
            vid_path = os.path.join(self.root_dir, 'leftImg8bit_sequence_trainvaltest', 'leftImg8bit_sequence', self._shortphase(), city_key, seq_key)
            vid = torch.load(vid_path + self.suffix)
            ann_path = os.path.join(self.root_dir, 'gtFine_trainvaltest', 'gtFine', self._shortpath(), city_key, seq_key)
            ann = torch.load(ann_path + self.suffix)

        if self.common_transform:
            vid = self.common_transform(vid)
            ann = self.common_transform(ann)
        if self.transform:
            vid = self.transform(vid)
        if self.label_transform:
            ann = self.label_transform(ann)
        return (vid, ann)


    def __len__(self):
       return len(self.dataset.get_ids())
