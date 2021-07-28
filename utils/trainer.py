from config import *
import torch
import torch.nn as nn
from utils import accuracy, AverageMeter
import time
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Trainer:
  def __init__(self, model):
    self.ce = nn.CrossEntropyLoss().cuda()
    self.mse = nn.MSELoss().cuda()
    self.opt = torch.optim.SGD(model.parameters(), lr, weight_decay=1e-4, momentum=0.9, nesterov=True)
    self.model = model
    self.print_freq=10

  def train(self, train_loader, epoch):
      batch_time = AverageMeter()
      data_time = AverageMeter()
      losses = AverageMeter()
      top1 = AverageMeter()
      top5 = AverageMeter()

      # set to train mode
      model.train()
    
      self.adjust_learning_rate(self.opt, epoch)

      end = time.time()
      for i, (input, target) in enumerate(train_loader):

          # dataloading duration
          data_time.update(time.time() - end)

          if use_gpu:
             input = torch.autograd.Variable(input.float().cuda())
             target = torch.autograd.Variable(target.long().cuda())

          else:
             input = torch.autograd.Variable(input.float())
             target = torch.autograd.Variable(target.long())

          # compute output
          output = self.model(input)
          loss = self.ce(output, target)

          #measure accuracy and record loss
          prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
          losses.update(loss.data, input.size(0))
          top1.update(prec1, input.size(0))
          top5.update(prec5, input.size(0))

          # compute gradient and do SGD step
          self.opt.zero_grad()
          loss.backward()
          #nn.utils.clip_grad_value_(model.parameters(), clip)
          self.opt.step()

          # measure elapsed time
          batch_time.update(time.time() - end)
          end = time.time()


          if i % self.print_freq == 0:
              print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                     self.epoch, i, len(train_loader), batch_time=batch_time,
                     data_time=data_time, loss=losses, top1=top1, top5=top5))


      results = open('cnn_train.txt', 'a')
      results.write('Epoch: [{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
             self.epoch, i, len(train_loader), loss=losses,
             top1=top1, top5=top5))
      results.close()

  def validate(self, val_loader):
      batch_time = AverageMeter()
      data_time = AverageMeter()
      losses = AverageMeter()
      top1 = AverageMeter()
      top5 = AverageMeter()

      # set to validation mode
      model.eval()

      end = time.time()
      with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            data_time.update(time.time() - end)
s
            if use_gpu:
               input = torch.autograd.Variable(input.float().cuda())
               target = torch.autograd.Variable(target.long().cuda())
            else:
               input = torch.autograd.Variable(input.float())
               target = torch.autograd.Variable(target.long())


            # compute output
            output = self.model(input)
            loss = self.ce(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if i % self.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                       top1=top1, top5=top5))

        print(' Epoch:{0} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(self.epoch, top1=top1, top5=top5))

        results = open('cnn_valid.txt', 'a')
        results.write('Epoch:{0} Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
              .format(self.epoch, loss=losses, top1=top1, top5=top5))
        results.close()

        return top1.avg

  def adjust_learning_rate(self, optimizer, epoch):
      """Drops the learning rate by 0.1 for every 10 epochs"""
      l_rate = lr * (lr_decay ** (epoch // 10))
      for param_group in optimizer.param_groups:
          param_group['lr'] = l_rate

