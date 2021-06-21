#!/usr/bin/env python3
'''
Download script for Objectron dataset (https://research.google/tools/datasets/objectron/).
Shortens the videos to save space.
'''
import argparse
import os
import requests
from random import sample
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


# info about the objectron dataset
public_url = 'https://storage.googleapis.com/objectron'
classes = ['bike', 'book', 'bottle', 'camera', 'cereal_box', 'chair', 'cup', 'laptop', 'shoe']

# command line argument parsing
parser = argparse.ArgumentParser(description='Download Objectron dataset and clip videos.')
parser.add_argument('-s', '--seconds', type=int, default=3, help='Number of seconds to clip each video to.')
parser.add_argument('--folder', type=str, default='objectron', help='Target folder to save data to.')
parser.add_argument('-c', '--classes', type=str, nargs='*', default=classes, help='Classes to restrict the download to.')
parser.add_argument('-f', '--force', action='store_true', help='Force new download of videos.')
parser.add_argument('--subset', type=str, default='train', choices=['train', 'test', 'all'], help='Data subset to download from (train, test or all).')
parser.add_argument('-n', '--num-vids-per-class', type=int, help='Number of videos per class to get.')


def get_vidnames(classes=classes, subset='train'):
    '''Downloads names of videos in dataset. Returns list of class-name pairs.'''
    vnames = []
    if subset == 'train':
        suffix = '_annotations_train'
    elif subset == 'test':
        suffix = '_annotations_test'
    elif subset == 'all':
        suffix = '_annotations'
    else:
        raise ValueError('Unknown subset: {}'.subset)
    # download index files for all classes
    for c in classes:
        names = requests.get(public_url + '/v1/index/' + c + suffix).text.split('\n')[:-1]
        for vname in names:
            vnames.append((c, vname))
    return vnames

def sample_vidnames(vnames, classes, vids_per_class):
    '''Randomly selects `vids_per_class` video names per class in `classes` from
    `vnames`.'''
    new_vnames = []
    for c in classes:
        filtered = list(filter(lambda pair: pair[0] == c, vnames))
        if vids_per_class is not None:
            filtered = sample(filtered, vids_per_class)
        new_vnames.extend(filtered)
    return new_vnames

def download_video(vidname, fname):
    url = public_url + '/videos/' + vidname + '/video.MOV'
    video = requests.get(url)
    with open(fname, 'wb') as f:
        f.write(video.content)

def clip_video(fname, tstart=0, tend=3):
    temp = os.path.dirname(fname) + 'cut.mov'
    ffmpeg_extract_subclip(fname, tstart, tend, targetname=temp)
    os.rename(temp, fname)     

def get_clip_video(vidname, fname, tstart=0, tend=3, force=False):
    if not os.path.exists(os.path.dirname(fname)): # create folder if it does not exist yet
        os.mkdir(os.path.dirname(fname))
    if force or not os.path.isfile(fname): # skip download if video is present already
        download_video(vidname, fname)
        clip_video(fname, tstart=tstart, tend=tend)

if __name__ == '__main__':
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        os.mkdir(args.folder)
    
    # get list of video names
    vnames = get_vidnames(classes=args.classes, subset=args.subset)
    vnames = sample_vidnames(vnames, args.classes, args.num_vids_per_class)
    
    # download all videos
    for i, (c, v) in enumerate(vnames):
        if not c in args.classes:
            continue
        print('Getting video {} of {}.'.format(i, len(vnames)), end='\r')
        objclass, batch, j = v.split('/')
        assert objclass == c
        get_clip_video(v, os.path.join(args.folder, c, batch + '-' + j + '.mov'),
            tstart=0, tend=args.seconds, force=args.force)

