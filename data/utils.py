from math import floor
import numpy as np
import torch
from torch.nn.functional import pad

def subsample_ids(nframes, nsamples):
    '''
    Computes the indices of frames for subsampling a video.
    '''
    step = floor(nframes / nsamples)
    return range(0,nsamples*step,step)

def subsample(video, nsamples):
    '''
    Subsamples a video to have the specified number of frames (nsamples).
    '''
    frames = video.size()[0]
    if frames >= nsamples:
        frame_ids = subsample_ids(frames, nsamples)
        return(video[frame_ids, :, :, :])
    else: # if there are less frames than we want samples, we need to zero-pad the video.
        return pad(video, (0, 0, 0, 0, 0, 0, 0, nsamples - frames))

class FlippingTransform():
    '''
    Transform to randomly flip whole videos (each frame alike).
    '''
    def __init__(self, p_flipping, flipping_axis=3, rng_state=42):
        '''
        Args
            p_flipping: Probability of flipping
            flipping_axis: Axis for which the flip should occur
            rng_state: RNG state for the internal random number generator
        '''
        self.p_flipping = p_flipping
        self.flipping_axis = flipping_axis
        self.rng = np.random.default_rng(rng_state)

    def _flip_video(self, vid):
        vid = torch.flip(vid, [min(self.flipping_axis,vid.ndim-1)])
        return vid
        #return torch.flip(vid, [self.flipping_axis])

    def __call__(self, vids):
        # choose whether to flip or not to flip
        if self.rng.random() > self.p_flipping:
            return vids
        # if we flip, apply transform to every video
        # if `vids` is a list, apply it to every video in the list
        # otherwise flip the variable itself (since it is just one video)
        if isinstance(vids, list):
            return [self._flip_video(vid) for vid in vids]
        return self._flip_video(vids)
