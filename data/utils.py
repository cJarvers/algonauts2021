from math import floor
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
