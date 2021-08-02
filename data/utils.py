from math import floor

def subsample_ids(nframes, nsamples):
    step = floor(nframes / nsamples)
    return range(0,nsamples*step,step)

def subsample(video, nsamples):
    '''
    Subsamples a video to have the specified number of frames (nsamples).
    '''
    nframes = video.size()[0]
    frame_ids = subsample_ids(nframes, nsamples)
    return(video[frame_ids, :, :, :])
