'''
Wrapper / loader for DAVIS dataset.
'''
import os
import math
import numpy as np
import random
import pathlib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from data.utils import subsample_ids

class DAVISDB():
    '''
    Underlying implementation for the DAVIS dataset.

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
        self._perform_train_val_split(root_dir)

    @classmethod
    def _get_subdirs(cls, dirpath):
        # cf. https://docs.python.org/3/library/pathlib.html#basic-use
        return [x for x in dirpath.iterdir() if x.is_dir()]

    @classmethod
    def _get_files_in_dir(cls, dirpath):
        return [x for x in dirpath.iterdir() if x.is_file()]

    @classmethod
    def _get_valid_videos(cls, vid_dirs, ann_dirs):
        '''
        Reduces the list of potential videos in the dataset to only the valid
        candidates, i.e. for which at least one video and the video exist.

        Args:
            *vid_dirs: List of potential videos in the dataset
            *ann_dirs: List of annotation videos in the dataset
        '''
        # the lists should be identical
        vid_dir_strs = [str(vid).split(pathlib.os.sep)[-1] for vid in vid_dirs]
        ann_dir_strs = [str(ann).split(pathlib.os.sep)[-1] for ann in ann_dirs]
        if len(vid_dir_strs) != len(ann_dir_strs) or (np.sort(vid_dir_strs) != np.sort(ann_dir_strs)).any():
            raise ValueError("Videos and annotations differ!")
        for (vid_dir, ann_dir) in zip(vid_dirs, ann_dirs):
            vid_img_strs = [vid_img.stem for vid_img in DAVISDB._get_files_in_dir(vid_dir)]
            ann_img_strs = [ann_img.stem for ann_img in DAVISDB._get_files_in_dir(ann_dir)]
            if len(vid_img_strs) == 0 or len(vid_img_strs) != len(ann_img_strs) or \
               (np.sort(vid_img_strs) != np.sort(ann_img_strs)).any():
                raise ValueError("Videos and annotations have different number of frames!")
        return (vid_dirs, ann_dirs)

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
        frame_0 = DAVISDB._read_image(frame_paths[0])
        vid = torch.zeros((len(frame_paths),*frame_0.shape),dtype=frame_0.dtype) # Tensor [T, H, W, C]
        vid[0,:] = frame_0
        for (frame_idx, frame_path) in enumerate(frame_paths[1:]):
            vid[frame_idx,:] = DAVISDB._read_image(frame_path)
        # yields Tensor[C, T, H, W]
        if len(vid.shape) == 4:
            vid = vid.movedim(3, 0)
        else:
            vid = torch.unsqueeze(vid,0)
        return vid

    @classmethod
    def _read_txt(cls, txt_file):
        '''
        Reads lines of a .txt file.

        Args:
            *txt_file: path to the txt file
        '''
        lines = []
        with open(str(txt_file), 'r') as f:
            for line in f:
                lines.append(line.split('\n')[0])
        return lines

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
        vids = Path(root_dir) / "JPEGImages" / "Full-Resolution"
        anns = Path(root_dir) / "Annotations" / "Full-Resolution"
        vid_dirs = DAVISDB._get_subdirs(vids)
        ann_dirs = DAVISDB._get_subdirs(anns)

        valid_vids, valid_anns = DAVISDB._get_valid_videos(vid_dirs, ann_dirs)

        self.videos = np.sort(valid_vids)
        self.annotations = np.sort(valid_anns)

    def _perform_train_val_split(self, root_dir):
        '''
        Read train/val split from files
        '''
        train_file = Path(root_dir) / "ImageSets" / "2017" / "train.txt"
        val_file = Path(root_dir) / "ImageSets" / "2017" / "val.txt"
        videos_train = np.array(DAVISDB._read_txt(train_file))
        videos_val = np.array(DAVISDB._read_txt(val_file))

        video_names = np.array([str(video).split(pathlib.os.sep)[-1] for video in self.videos])
        self.train_ids = np.concatenate([np.where(video == video_names)[0] for video in videos_train])
        self.validation_ids = np.concatenate([np.where(video == video_names)[0] for video in videos_val])

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

    def _get_video(self, video_dir):
        '''
        Returns a properly subsampled video stemming from the given `video_dir`.

        Args:
            *video_dir: path to the directory where images are residing
        '''
        # select which frames to read from the video
        frames = DAVISDB._get_files_in_dir(video_dir)
        subsampled_frame_ids = subsample_ids(len(frames), self.nsamples)
        sampled_frames = [frames[idx] for idx in subsampled_frame_ids]

        return DAVISDB._frames_to_video(sampled_frames)

    def get_sample(self, sample_idx):
        '''
        Returns a random, properly subsampled video belonging to the person as
        indicated by the `sample_idx`.

        Args:
            *sample_idx (int): index indicating a person in the dataset
        '''
        video = self._get_video(self.videos[sample_idx])
        annotation = self._get_video(self.annotations[sample_idx])

        return (video, annotation)

class DAVISDataset(Dataset):
    '''
    Wrapper for the DAVIS dataset.
    Args:
        *root_dir (str): Directory from which to load data. Should contain a
                         folder `frame_images_DB` 
                         as well as subfolders with training/validation data.
        *phase (str): `training` or `validation`
        *nframes (int): number of frames to subsample every video to
        *transform: PyTorch transform to apply to the videos/frames
        *rng_state (int): initial state of the random number generator
    '''
    def __init__(self, root_dir, phase, nframes, common_transform, augmentation_transform, rng_state=0):
        self.root_dir = root_dir
        self.phase = phase
        self.nframes = nframes
        self.common_transform = common_transform
        self.augmentation_transform = augmentation_transform
        self.dataset = DAVISDB(root_dir, phase, nframes, rng_state=rng_state)
        # TODO: compute or load mean and standard deviation over complete dataset
        #       to normalize videos
        #       Alternatively, we can add a batch-norm layer at the front of the network

    def __getitem__(self, idx):
        (vid, ann) = self.dataset.get_sample(idx)

        if self.common_transform:
            vid = self.common_transform(vid)
            ann = self.common_transform(ann)
        if self.augmentation_transform:
            vid = self.augmentation_transform(vid)
        return (vid, ann)


    def __len__(self):
       return len(self.dataset.get_ids())
