from __future__ import print_function, division
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import matplotlib.pyplot as plt
import random
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import time
import torchsummary
import cv2
from loss_F import GIoU,IoU,objectness_loss,obj_GIoU,negative_IoU
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import WRutils

def init_process(rank,size,fn,backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '35623'
    dist.init_process_group(backend,rank=rank,world_size = size)
    fn(rank,size)

date = '1104'
model_N = '100_GIOU'
folder = '22{}_crop_model{}'.format(date,model_N)
path = "train_data/1030_nonMotion"
train_path = "train_data/1030_nonMotion/labels"
save_path = '{}_result'.format(date)

#create new folder
modelpath = "{}/model{}".format(save_path,folder)
trainpath = "{}/train{}".format(save_path,folder)
valipath = "{}/vali{}".format(save_path,folder)
if not os.path.exists(modelpath):
    os.makedirs(modelpath)
    os.makedirs(trainpath)
    os.makedirs(valipath)

learning_rate = 1
# parameters
image_size = 100
learning_rate = 0.0001
training_epochs = 2000
batch_size = 512

## image dataset
labels_list = os.listdir(train_path)
n = len(labels_list)



portion_vali = 0.05
num_vali = int(n * portion_vali)
num_train = int((n - num_vali)/4)
num_train_ = int(n - num_vali - (3*num_train))


    
train_set = WRutils.Anafidataset(root_dir=path)

train_0, train_1, train_2, train_3, vali = torch.utils.data.random_split(train_set, [num_train, num_train, num_train, num_train_, num_vali])


train_dataset_0 = torch.utils.data.DataLoader(dataset=train_0,
                                           batch_size=batch_size,
                                           num_workers=8, 
                                           shuffle=True,
                                           drop_last=False,
                                           pin_memory = True,
                                           persistent_workers = True)
train_dataset_1 = torch.utils.data.DataLoader(dataset=train_1,
                                           batch_size=batch_size,
                                           num_workers=8, 
                                           shuffle=True,
                                           drop_last=False,
                                           pin_memory = True,
                                           persistent_workers = True)
train_dataset_2 = torch.utils.data.DataLoader(dataset=train_2,
                                           batch_size=batch_size,
                                           num_workers=8, 
                                           shuffle=True,
                                           drop_last=False,
                                           pin_memory = True,
                                           persistent_workers = True)
train_dataset_3 = torch.utils.data.DataLoader(dataset=train_3,
                                           batch_size=batch_size,
                                           num_workers=8, 
                                           shuffle=True,
                                           drop_last=False,
                                           pin_memory = True,
                                           persistent_workers = True)
vali_dataset = torch.utils.data.DataLoader(dataset=vali,
                                           batch_size=batch_size,
                                           num_workers=8, 
                                           shuffle=True,
                                           drop_last=False,
                                           pin_memory = True,
                                           persistent_workers = True)
datasets = [train_dataset_0, train_dataset_1, train_dataset_2, train_dataset_3]




def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
        
        
def run(rank, size):
    pre_path = '1'
    pre_best_path = '1'
    best_IoU = 0.1
    torch.manual_seed(1234)
    train_set = datasets[rank]
    model = DDP(WRutils.Anafi_model(), device_ids = [rank])
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    pre_T = time.time()
    for epoch in range(training_epochs):
        avg_cost = 0
        vali_avg_cost=0
        train_mIOU = 0
        vali_mIOU = 0
        train_IOU = 0
        vali_IOU = 0
        model.train()
        for X, Y in train_set:
            optimizer.zero_grad()
            hypothesis = model(X)
            YY = hypothesis.detach().to('cpu')
            cost = obj_GIoU(hypothesis,Y.cuda())
            cost.backward()
            average_gradients(model)
            optimizer.step()
            Y = Y.detach().to('cpu')
            train_IOU, num = negative_IoU(YY, Y)
            train_mIOU += train_IOU / int(40914*(1-portion_vali))
            avg_cost += cost / int(40914*(1-portion_vali))
            # 162912 for 1026 total data
        model.eval()
        for X, Y in vali_dataset:
            with torch.no_grad():
                vali_X = X.cuda()
                vali_Y = Y.cuda()
                vali_hypothesis = model(vali_X)
                vali_YY = vali_hypothesis.detach().to('cpu')
                vali_cost = obj_GIoU(vali_hypothesis, vali_Y)           
                vali_IOU, vali_num = negative_IoU(vali_YY, vali_Y.detach().to('cpu'))
                vali_mIOU += vali_IOU / (int(40914*portion_vali))
                vali_avg_cost += vali_cost / int(40914*portion_vali)
        print('{} [{:>4}] cost= [{:>.5} / {:>.5}] / IOU= [{:>.5} / {:>.5}] / time= {:>.4}, ETA= {:>.4}h'.format(model_N,epoch + 1, avg_cost, vali_avg_cost,train_mIOU,vali_mIOU,time.time()-pre_T, (time.time()-pre_T)*(training_epochs-epoch)/3600))
        pre_T = time.time()
        
        torch.save(model.module.state_dict(), "{}/model{}/Last_epoch{}_cost{:>.5}.pth".format(save_path,folder,epoch+1,avg_cost))
        if os.path.isfile(pre_path):
            os.remove(pre_path)
        pre_path = "{}/model{}/Last_epoch{}_cost{:>.5}.pth".format(save_path,folder,epoch+1,avg_cost)
        if best_IoU <= vali_mIOU:
            torch.save(model.module.state_dict(), "{}/model{}/Best_epoch{}_cost{:>.5}.pth".format(save_path,folder,epoch+1,avg_cost))        
            if os.path.isfile(pre_best_path):
                os.remove(pre_best_path)
            best_IoU = vali_mIOU
            pre_best_path = "{}/model{}/Best_epoch{}_cost{:>.5}.pth".format(save_path,folder,epoch+1,avg_cost)
        
    print('Learning Finished!')
    
def main():
    world_size = 4
    processes = []
    for rank in range(world_size):
        p = mp.Process(target = init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

#    mp.spawn(run,
 #            args=(world_size,),
  #           nprocs=world_size,
   #          join=True)


if __name__ == '__main__':
    main()
