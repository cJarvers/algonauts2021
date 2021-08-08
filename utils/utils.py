import os
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    '''
    Logs the training loss and performs checkpointing for multiGPU training.
    
    Args:
        *ckptpath (str or path): base path at which to save checkpoints
        *logpath (str or path): base path at which to save log files
        *logevery (int): logging interval (in number of batches)
    '''
    def __init__(self, ckptpath, logpath, logevery=1000):
        self.losscurve = []
        self.metriccurve = []
        self.ckptpath = ckptpath
        self.logpath = logpath
        self.logevery = logevery
        self.lastlog = 0
        if not(os.path.exists(os.path.dirname(self.ckptpath))):
            os.makedirs(os.path.dirname(self.ckptpath))
        if not(os.path.exists(os.path.dirname(self.logpath))):
            os.makedirs(os.path.dirname(self.logpath))
        pass
        
    def log(self, epoch, batch, loss, metric, model_dict, decoder_dict, rank):
        self.losscurve.append((epoch, batch, loss))
        self.metriccurve.append((epoch, batch, metric))
        # if the current epoch is a multiple of the 'logevery' interval, write checkpoints and logs to file
        if batch - self.lastlog >= self.logevery:
            torch.save({'loss': self.losscurve, 'metric': self.metriccurve}, self.logpath + f'rank{rank}.log')
            torch.save(decoder_dict, self.ckptpath + f'decoder_{rank}_b{batch}.ckpt')
            torch.save(model_dict, self.ckptpath + f'model_{rank}_b{batch}.ckpt')    
            self.lastlog = batch
            
        

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
