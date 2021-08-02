from math import floor
from torch.nn.functional import pad

def subsample(video, nsamples):
    '''
    Subsamples a video to have the specified number of frames (nsamples).
    '''
    frames = video.size()[0]
    if frames >= nsamples:
        step = floor(frames / nsamples)
        return(video[0:nsamples*step:step, :, :, :])
    else: # if there are less frames than we want samples, we need to zero-pad the video.
        return pad(video, (0, 0, 0, 0, 0, 0, 0, nsamples - frames))
