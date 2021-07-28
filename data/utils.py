from math import floor


def subsample(video, nsamples):
    '''
    Subsamples a video to have the specified number of frames (nsamples).
    '''
    frames = video.size()[0]
    step = floor(frames / nsamples)
    return(video[0:nsamples*step:step, :, :, :])
