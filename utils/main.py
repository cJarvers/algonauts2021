import shutil
from config import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from trainer import Trainer
from resnet3d import resnet50
from dataset_ucf import UCFDataset, ToTensor
from torchsummary import summary
import time
import copy
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def save_checkpoint(state_dict, is_best, filename='./save_dir/vgg16_current_model.pth.tar'):
    torch.save(state_dict, filename)
    if is_best:
        shutil.copyfile(filename, trained_model_path_best)

def main():
    best_prec1 = 0       
    model = resnet50(pretrained=False)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        #   switch to parallelism
        #model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    
    #prepare dataset
    train_dataset = UCFDataset(root_dir=ucf_data_root, phase='train', transform=transforms.Compose([ToTensor()]))                                                                                                                                                                          
    valid_dataset = UCFDataset(root_dir=ucf_data_root, phase='test', transform=transforms.Compose([ToTensor()]))
                                                                                                                                                                          
    # Loading dataset into dataloader
    train_loader =  torch.utils.data.DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True,num_workers=num_workers)                                        
    val_loader =  torch.utils.data.DataLoader(valid_dataset,batch_size=test_batch_size,shuffle=True,num_workers=num_workers)
                                                  
    #start time for training
    start_time= time.time()

    trainer = Trainer(model)
    prec1=0
    for epoch in range(0, num_epochs):
        # train on train dataset
        trainer.train(train_loader, epoch)

        # evaluate on validation set for every 3 epochs
        if (epoch+1)%3==0:
          prec1 = trainer.validate(val_loader, epoch)
          print('Top Precision:',prec1)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(model.state_dict(), is_best, trained_model_path)

    end_time = time.time()
    duration= (end_time - start_time)/3600
    print("Duration:")
    print(duration)

if __name__ == '__main__':
   main()
