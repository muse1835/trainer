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
from WRutils import Anafi_model, Anafidataset

pltmode = 0
stack_loss = []
stack_vali_loss = []
stack_epoch = []
pre_T = 0
plot_mode = 0
pre_path = '1'
pre_best_path = '1'
best_IoU = 0.1
device = torch.device('cuda:0')
torch.cuda.set_device(device)
pos_num = 4323
# neg = 43321
for seed in range(778,779):  
    date = '0110'
    model_N = '100_GIOU'
    folder = '23{}_crop_model{}'.format(date,model_N)
    path = "train_data/nonblur"
    train_path = path + "/labels"
    save_path = '{}_result'.format(date)
    
    #create new folder
    modelpath = "{}/model{}".format(save_path,folder)
    trainpath = "{}/train{}".format(save_path,folder)
    valipath = "{}/vali{}".format(save_path,folder)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
        os.makedirs(trainpath)
        os.makedirs(valipath)
        
    start = time.time()  # 시작 시간 저장


    torch.cuda.manual_seed(seed)

        
    # parameters
    image_size = 100
    learning_rate = 0.0001
    training_epochs = 2000
    batch_size = 512
    
    ## image dataset
    labels_list = os.listdir(train_path)
#    negative_frame = pd.read_csv('{}/data.csv'.format(negative_path))
    n = len(labels_list)

    # portion_vali = 0.3
    # portion_test = 0.001
    # num_vali = int(n * portion_vali)
    # num_test = int(n * portion_test)
    # num_train = n - num_vali
    
    portion_vali = 0.05
    num_vali = int(n * portion_vali)
    num_train = n - num_vali
    
#        num_train = int(n * portion_train)
#        num_dummy = n - num_vali  - num_train
    
#    negative_dummy = len(negative_frame)- int(0.1*len(negative_frame))

        
        
    train_set = Anafidataset(root_dir=path)
#    negative_set = Balldataset(csv_file='{}/data.csv'.format(negative_path),
#                                    root_dir=negative_path)
    
    train, vali = torch.utils.data.random_split(train_set, [num_train, num_vali])
#    negative, _ = torch.utils.data.random_split(negative_set, [int(0.1*len(negative_frame)),negative_dummy ])
    
    
    train_dataset = torch.utils.data.DataLoader(dataset=train,
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
#    negative_dataset = torch.utils.data.DataLoader(dataset=negative,
#                                               batch_size=1,
#                                               shuffle=True,
#                                               drop_last=False)
#                                               pin_memory = True)

    train_plot_index = np.zeros(5)
    vali_plot_index = np.zeros(5)
    for i in range (5):
        train_plot_index[i] = train_dataset.dataset.indices[int(random.randint(0,num_train-1))]
        vali_plot_index[i] =  vali_dataset.dataset.indices[int(random.randint(0,num_vali-1))]
#    train_list = []
#    for i in range(len(train_dataset.dataset.indices)):
#        train_list = np.append(train_list, train_balls_frame.iloc[train_dataset.dataset.indices[i],4])
#    train_list = list(train_list)
#    val_list = []
#    for i in range(len(vali_dataset.dataset.indices)):
#        val_list = np.append(val_list, train_balls_frame.iloc[vali_dataset.dataset.indices[i],4])
#    val_list = list(val_list)
#    np.save("RP_result/model{}".format(folder), train_list)
#    np.save("RP_result/model{}".format(folder), val_list)
    
        
        
    # instantiate CNN model
    # model = nn.DataParallel(CNN().cuda(), device_ids=[2,3])
    # CNN().cuda()
    model = Anafi_model().cuda()
    model = nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()
    # model = DDP(model,device_ids=[0,1,2,3]).cuda()
    # model.load_state_dict(torch.load('{}/result/model{}/cost161_0.0019307868788018823.pth'.format(train_path,folder)))
#    model = nn.DataParallel(model)
    # define cost/loss & optimizer
    # criterion = torch.nn.MSELoss().cuda()    # Softmax is internally computed.
    # criterion = nn.MSELoss()
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    # train my model
    total_batch = len(train_dataset.dataset.indices)
    # torch.backends.cudnn.benchmark = True
    torchsummary.summary(model,(3,image_size,image_size))

    print('Learning started. It takes sometimes.')

    for epoch in range(training_epochs):
        avg_cost = 0
        vali_avg_cost=0
        train_mIOU = 0
        vali_mIOU = 0
        train_IOU = 0
        vali_IOU = 0
        train_count = 0
        vali_count = 0
        model.train()
        for X, Y in train_dataset:
            X = X.cuda()
            Y = Y.cuda()
            optimizer.zero_grad()
            hypothesis = model(X)
            YY = hypothesis.detach().to('cpu')
            cost = obj_GIoU(hypothesis,Y)
            cost.backward()
            optimizer.step()
            Y = Y.detach().to('cpu')
            train_IOU, num = negative_IoU(YY, Y)
            train_mIOU += train_IOU / int(pos_num*(1-portion_vali))
            avg_cost += cost / int(pos_num*(1-portion_vali))
            # 162912 for 1026 total data
            del hypothesis
            del cost
        model.eval()
        for X, Y in vali_dataset:
            with torch.no_grad():
                vali_X = X.cuda()
                vali_Y = Y.cuda()
                vali_hypothesis = model(vali_X)
                vali_YY = vali_hypothesis.detach().to('cpu')
                vali_cost = obj_GIoU(vali_hypothesis, vali_Y)           
                vali_IOU, vali_num = negative_IoU(vali_YY, vali_Y.detach().to('cpu'))
                vali_mIOU += vali_IOU / (int(pos_num*portion_vali))
                vali_avg_cost += vali_cost / int(pos_num*portion_vali)
        
        if avg_cost < 10:
            pltmode = 0
        if pltmode == 1:
            stack_loss = np.append(stack_loss,avg_cost.detach().to('cpu'))
            stack_vali_loss = np.append(stack_vali_loss, vali_avg_cost.detach().to('cpu'))
            stack_epoch = np.append(stack_epoch, epoch)
            plt.plot(stack_epoch, stack_loss)
            plt.plot(stack_epoch, stack_vali_loss)
            plt.show()

        print('{} [{:>4}] cost= [{:>.5} / {:>.5}] / IOU= [{:>.5} / {:>.5}] / time= {:>.4}, ETA= {:>.4}h'.format(model_N,epoch + 1, avg_cost, vali_avg_cost,train_mIOU,vali_mIOU,time.time()-pre_T, (time.time()-pre_T)*(training_epochs-epoch)/3600))
        pre_T = time.time()

    
    ##1008 del
        if plot_mode == 1:
            train_imgs = []
            vali_imgs = []
            for i in range(len(train_plot_index)):
                plot_Y = train_balls_frame.iloc[int(train_plot_index[i]),:4]
                # if sum(plot_Y[:2]) != 0:
                plot_Y[:4] = plot_Y[:4]/(200/image_size)
                train_X = cv2.imread("{}/{}".format(train_path,train_balls_frame.iloc[int(train_plot_index[i]),4]))
                train_X = cv2.resize(train_X,(image_size,image_size), interpolation=cv2.INTER_AREA)
                train_X_tensor = torch.cuda.FloatTensor(train_X.transpose(2,0,1))
    #            train_X_tensor = torch.cuda.FloatTensor(train_X.transpose(2,0,1))
                train_Y = model(train_X_tensor.reshape([1,3,image_size,image_size]))
                train_Y = train_Y.detach().to('cpu')
                train_Y = train_Y.reshape([5])
                train_Y[:4] = train_Y[:4]*image_size
    #                train_iou = iou_1(train_Y, plot_Y)
    #                cv2.putText(train_X, "IOU = {:>.4}".format(float(train_iou)), (1,5), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0,255),1)
                cv2.circle(train_X,(int(train_Y[0]),int(train_Y[1])),3,(255,255,255))
                cv2.circle(train_X,(int(plot_Y[0]),int(plot_Y[1])),3,(0,255,255))
                cv2.rectangle(train_X,
                              (int(train_Y[0]-0.5*train_Y[2]),int(train_Y[1]-0.5*train_Y[3])),
                              (int(train_Y[0]+0.5*train_Y[2]),int(train_Y[1]+0.5*train_Y[3])),(255,255,255),1)
                cv2.rectangle(train_X,
                              (int(plot_Y[0]-0.5*plot_Y[2]),int(plot_Y[1]-0.5*plot_Y[3])),
                              (int(plot_Y[0]+0.5*plot_Y[2]),int(plot_Y[1]+0.5*plot_Y[3])),(0,0,255),1)
                train_X  = cv2.resize(train_X,(200,200), interpolation=cv2.INTER_AREA)
                if i == 0:
                    train_imgs = train_X
                else:
                    train_imgs = np.concatenate((train_imgs,train_X),axis=1)
            for i in range(len(vali_plot_index)):
                refer_Y_vali = train_balls_frame.iloc[int(vali_plot_index[i]),:4]
                # if sum(refer_Y_vali[:2]) != 0:
                refer_Y_vali[:4] = refer_Y_vali[:4]/(200/image_size)
                plot_vali_X = cv2.imread("{}/{}".format(train_path,train_balls_frame.iloc[int(vali_plot_index[i]),4]))
                plot_vali_X = cv2.resize(plot_vali_X,(image_size,image_size), interpolation=cv2.INTER_AREA)
                plot_vali_X_tensor = torch.cuda.FloatTensor(plot_vali_X.transpose(2,0,1))
                plot_vali_Y = model(plot_vali_X_tensor.reshape([1,3,image_size,image_size]))
                plot_vali_Y = plot_vali_Y.to('cpu')
                plot_vali_Y = plot_vali_Y.reshape([5])
                plot_vali_Y[:4] = plot_vali_Y[:4] * image_size
    #                vali_iou = iou_1(plot_vali_Y, refer_Y_vali)
    #                cv2.putText(plot_vali_X, "IOU = {:>.4}".format(float(vali_iou)), (1,5), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0,255),1)
                cv2.circle(plot_vali_X,(int(plot_vali_Y[0]),int(plot_vali_Y[1])),3,(255,255,255))
                cv2.circle(plot_vali_X,(int(refer_Y_vali[0]),int(refer_Y_vali[1])),3,(0,255,255))
                cv2.rectangle(plot_vali_X,
                              (int(plot_vali_Y[0]-0.5*plot_vali_Y[2]),int(plot_vali_Y[1]-0.5*plot_vali_Y[3])),
                              (int(plot_vali_Y[0]+0.5*plot_vali_Y[2]),int(plot_vali_Y[1]+0.5*plot_vali_Y[3])),(255,255,255),1)
                cv2.rectangle(plot_vali_X,
                              (int(refer_Y_vali[0]-0.5*refer_Y_vali[2]),int(refer_Y_vali[1]-0.5*refer_Y_vali[3])),
                              (int(refer_Y_vali[0]+0.5*refer_Y_vali[2]),int(refer_Y_vali[1]+0.5*refer_Y_vali[3])),(0,0,255),1)
                plot_vali_X = cv2.resize(plot_vali_X,(200,200), interpolation=cv2.INTER_AREA)
                if i == 0:
                    vali_imgs = plot_vali_X
                else:
                    vali_imgs = np.concatenate((vali_imgs,plot_vali_X),axis=1)
            total_imgs = np.concatenate((train_imgs,vali_imgs),axis=0)
            total_imgs = cv2.resize(total_imgs,(1280,720), interpolation=cv2.INTER_AREA)
            
            cv2.imshow('{}'.format(model_N),total_imgs)
            cv2.waitKey(10)           
            cv2.imwrite('{}/{}_{}_{}train.jpg'.format(trainpath,i, epoch +1, avg_cost),total_imgs)

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
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    
