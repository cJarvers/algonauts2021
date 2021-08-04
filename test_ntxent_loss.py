# quick and simple test script

print("Starting testing of YouTubeFaces dataloader")

import data.youtube_faces_loader as ytf
import utils.losses as ls
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ConvertImageDtype, Resize, Compose

print("Imported module")

def test_dataloader():
    print("Instantiating transform")
    transform = Compose([ConvertImageDtype(torch.float32), Resize((224, 224))])
    print("Instantiated transform")
    print("Instantiating dataloader")
    dl = DataLoader(ytf.YouTubeFacesDataset('/data/YouTubeFaces','training', 10, transform),
           batch_size=32, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=2, collate_fn=None,
           pin_memory=False, drop_last=True, timeout=0,
           worker_init_fn=ytf.YouTubeFacesDataset.worker_init_fn, prefetch_factor=2,
           persistent_workers=False)
    print("Instantiated dataloader")
    print("Iterating dataloader")
    it = 0
    for vl in dl:
        print("----")
        print("Iteration: {}".format(it))
        print("Shape: {}".format(vl[0].shape))
        print("Sample: {}".format(vl[1]))
        it += 1
    print("---")
    print("Iterated dataloader")


def test_loss():
    print("Instantiating transform")
    transform = Compose([ConvertImageDtype(torch.float32), Resize((224, 224))])
    print("Instantiated transform")
    print("Instantiating dataloader")
    dl = DataLoader(ytf.YouTubeFacesDataset('/data/YouTubeFaces','training', 10, transform),
           batch_size=32, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=2, collate_fn=None,
           pin_memory=False, drop_last=True, timeout=0,
           worker_init_fn=ytf.YouTubeFacesDataset.worker_init_fn, prefetch_factor=2,
           persistent_workers=False)
    print("Instantiated dataloader")
    print("Instantiating loss")
    loss = ls.NT_Xent(0.1)
    print("Instantiated loss")
    print("Iterating dataloader")
    it = 0
    for vl in dl:
        print("----")
        print("Iteration: {}".format(it))
        print("Shape: {}".format(vl[0].shape))
        print("Sample: {}".format(vl[1]))
        it += 1
        if it >= 1:
            break
    print("---")
    print("Iterated dataloader")
    print("Simulating embedding")
    print("X shape: {}".format(vl[0].shape))
    emb = torch.reshape(vl[0],(vl[0].shape[0],-1))
    print("Embedding shape: {}".format(emb.shape))
    print("Simulated embedding")
    print("Performing loss computation")
    l = loss(emb, vl[1])
    print("Loss: {}".format(l))
    print("Performed loss computation")




if __name__ == '__main__':
#    test_dataloader()
    test_loss()
