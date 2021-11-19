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
from loss_F import GIoU,IoU

plot_mode = 1
pre_T = 0
pltmode = 0
plot_mode = 1
pre_path = '1'
pre_best_path = '1'
best_IoU = 0.1
stack_loss = []
stack_vali_loss = []
stack_epoch = []
# import math
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"7
#torch.multiprocessing.get_context('spawn')
device = torch.device('cuda:0')
torch.cuda.set_device(device)

for seed in range(778,779):  
    date = '1115'
    model_N = '720p_GIOU'
    folder = '21{}_full_model{}'.format(date,model_N)
    path = "total_0422/full"
    train_path = 'total_0422/full'
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
    
    # plt.ion()  
    # device = torch.device("cuda")
    
    # for reproducibility
    torch.cuda.manual_seed(seed)
    # if device == 'cuda':
    #     torch.cuda.manual_seed_all(777)
        
        
        # parameters
#    image_size = 400
    img_width = 1280
    img_height = 720
    learning_rate = 0.0001
    training_epochs = 2000
    batch_size = 128
    
    ## image dataset
    train_balls_frame = pd.read_csv('{}/data_0422.csv'.format(train_path))
#    vali_balls_frame = pd.read_csv('{}/data.csv'.format(vali_path))
    n = len(train_balls_frame)
    # img_name = balls_frame.iloc[n, 4]
    # Y_data = balls_frame.iloc[n, :4]
    # Y_data = np.asarray(Y_data)
    # Y_data = Y_data.astype('float').reshape(-1, 4)
    
    portion_vali = 0.1
    num_vali = int(n * portion_vali)
    num_train = n - num_vali
    
#        portion_vali = 0.1
#        portion_train = 0.3
#        num_vali = int(n * portion_vali)    
#        num_train = int(n * portion_train)
#        num_dummy = n - num_vali -  num_train
    
    
    class Balldataset(Dataset):
        """Face Landmarks dataset."""
    
        def __init__(self, csv_file, root_dir, transform=None):
            """
            Args:
                csv_file (string): csv 파일의 경로
                root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
                transform (callable, optional): 샘플에 적용될 Optional transform
            """
            self.balls_frame = pd.read_csv('{}/data_0422.csv'.format(root_dir))
            self.root_dir = root_dir
            self.transform = 'transform'
            
        def __len__(self):
            # print(len(self.balls_frame))
            return len(self.balls_frame)
    
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
    
            # img_name = os.path.join(self.root_dir,
            #                         self.balls_frame.iloc[idx, 4])
            image = io.imread('{}/{}'.format(self.root_dir,self.balls_frame.iloc[idx, 4]))
#                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,(img_width,img_height), interpolation=cv2.INTER_AREA)
            image = image.reshape(img_height,img_width,3)
            image = image.transpose(2,0,1)
    
#            print(image.shape)
            # image = image[0, :, :]
            # image = image.reshape(1,image_size,image_size)
            # image = image.reshape(1,image_size,image_size)
            # image[0,:,:] = (image[0,:,:] - np.min(image[0,:,:]))/ (np.max(image[0,:,:]) - np.min(image[0,:,:]))
            # image[1,:,:] = (image[1,:,:] - np.min(image[1,:,:]))/ (np.max(image[1,:,:]) - np.min(image[1,:,:]))
            # image[2,:,:] = (image[2,:,:] - np.min(image[2,:,:]))/ (np.max(image[2,:,:]) - np.min(image[2,:,:]))
            # image[0,:,:] = (image[0,:,:] - np.mean(image[0,:,:]))/ np.std(image[0,:,:])
            # image[1,:,:] = (image[1,:,:] - np.mean(image[1,:,:]))/ np.std(image[1,:,:])
            # image[2,:,:] = (image[2,:,:] - np.mean(image[2,:,:]))/ np.std(image[2,:,:])
            balls = self.balls_frame.iloc[idx, :4]/10
            balls[0] = balls[0]/128
            balls[1] = balls[1]/72
            balls[2] = balls[2]/128
            balls[3] = balls[3]/72
            balls = np.array([balls])
            balls = balls.astype('float').reshape(4)
            # sample = {'image': image, 'balls': balls}
#            dummy = np.ones([3,72,128])
#            X = torch.cuda.FloatTensor(dummy)
            # X = X.cuda()
            # Y = Y.cuda()
            # Y = torch.tensor(balls, dtype=torch.long, device=device)
            # if self.transform:
            #     sample = self.transform(sample)
    
            # return sample
            return torch.FloatTensor(image), torch.FloatTensor(balls)
        
    train_set = Balldataset(csv_file='{}/data_0422.csv'.format(train_path),
                                    root_dir=train_path)

    
    train, vali = torch.utils.data.random_split(train_set, [num_train, num_vali])
    # train, vali, test, dummy = torch.utils.data.random_split(ball_dataset, [num_train, num_vali, num_test, num_dummy])
    
    
    train_dataset = torch.utils.data.DataLoader(dataset=train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers = 16,
                                              drop_last=False,
                                              pin_memory = True)
    vali_dataset = torch.utils.data.DataLoader(dataset=vali,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=False,
                                              num_workers = 16,
                                              pin_memory = True)
    # test_data = torch.utils.data.DataLoader(dataset=test,
    #                                           batch_size=1,
    #                                           shuffle=True,
    #                                           drop_last=False)
                                               # pin_memory = True)
    train_plot_index = np.zeros(4)
    vali_plot_index = np.zeros(4)
    for i in range (4):
        train_plot_index[i] = train_dataset.dataset.indices[int(random.randint(0,num_train-1))]
        vali_plot_index[i] = vali_dataset.dataset.indices[int(random.randint(0,num_vali-1))]
    train_list = []
    for i in range(len(train_dataset.dataset.indices)):
        train_list = np.append(train_list, train_balls_frame.iloc[train_dataset.dataset.indices[i],4])
    train_list = list(train_list)
    val_list = []
    for i in range(len(vali_dataset.dataset.indices)):
        val_list = np.append(val_list, train_balls_frame.iloc[vali_dataset.dataset.indices[i],4])
    val_list = list(val_list)
    np.save("{}/model{}".format(save_path,folder), train_list)
    np.save("{}/model{}".format(save_path,folder), val_list)
    

    # CNN Model (2 conv layers)
    class RPN(torch.nn.Module):
    
        def __init__(self):
            super(RPN, self).__init__()
            # L1 ImgIn shape=(?, 28, 28, 1)
            #    Conv     -> (?, 28, 28, 32)
            #    Pool     -> (?, 14, 14, 32)
            self.layer1 = torch.nn.Sequential(
                torch.nn.BatchNorm2d(3),
                torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=(3,4), stride=(3,4), padding=0),
                torch.nn.BatchNorm2d(4),
                torch.nn.Dropout(0.2),
    
                torch.nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=(3,4), stride=(3,4), padding=0),
                torch.nn.BatchNorm2d(16),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=(3,4), stride=(3,4), padding=0),
                torch.nn.BatchNorm2d(32),
                torch.nn.Dropout(0.2),
                
                
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=(3,4), stride=(3,4), padding=0),
                torch.nn.BatchNorm2d(64),
                torch.nn.Dropout(0.2),
                
                
                torch.nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=(3,4), stride=(3,4), padding=0),
                torch.nn.BatchNorm2d(16),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(kernel_size=(3,4), stride=(3,4), padding=0),
                torch.nn.BatchNorm2d(32),
                torch.nn.Dropout(0.2),
    
                torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=(3,4), stride=(3,4), padding=0),
                torch.nn.BatchNorm2d(16),
                torch.nn.Dropout(0.2),
    
                torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(32),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(64),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(32),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(64),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(kernel_size=(3,4), stride=(3,4), padding=0),
                torch.nn.BatchNorm2d(128),
                torch.nn.Dropout(0.2),
      
                torch.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(32),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(64),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(32),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(16),
                torch.nn.Dropout(0.2),
            
                torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(32),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(64),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(32),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(64),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(kernel_size=5, stride=5, padding=0),
                torch.nn.BatchNorm2d(128),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(64),
                torch.nn.Dropout(0.2),
                
                torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(32),
                torch.nn.Dropout(0.2), 
                
                torch.nn.Conv2d(32, 4, kernel_size=2, stride=2, padding=0)).cuda()
    
            # Final FC 7x7x64 inputs -> 10 outputs
    #            self.fc = torch.nn.Linear(784, 3, bias=True).cuda()
            # torch.nn.init.xavier_uniform_(self.fc.weight).cuda()
    
        def forward(self, x):
            out = self.layer1(x).cuda()
    #             out = torch.mean(x.view(x.size(0), x.size(1), 1,1),dim=2)
    #            print(out.size)
            out = out.view(-1, 4).cuda()   # Flatten them for FC
    #            out = self.fc(out)
            return out
        
        
    # instantiate CNN model
    # model = nn.DataParallel(CNN().cuda(), device_ids=[2,3])
    # CNN().cuda()
    model = RPN().cuda()
    model = nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()

#    model.eval()
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    # train my model
    torchsummary.summary(model,(3,img_height,img_width))

    total_batch = len(train_dataset.dataset.indices)
    print('Learning started. It takes sometime.')
    # torch.backends.cudnn.benchmark = True
    for epoch in range(training_epochs):
        avg_cost = 0
        vali_avg_cost=0
        train_mIOU = 0
        vali_mIOU = 0
        model.train()
        for X, Y in train_dataset:
            X = X.cuda()
            Y = Y.cuda()
            optimizer.zero_grad()
            hypothesis = model(X)
#                cost = criterion(hypothesis, Y)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()
            avg_cost += cost / len(train_dataset.dataset.indices)
            del hypothesis
            del cost
    
        model.eval()
        for X, Y in vali_dataset:
            with torch.no_grad():
                vali_X = X.cuda()
                vali_Y = Y.cuda()
                vali_hypothesis = model(vali_X)
#                    vali_cost = criterion(vali_hypothesis, vali_Y)
                vali_cost = criterion(vali_hypothesis, vali_Y)
                vali_avg_cost += vali_cost / len(vali_dataset.dataset.indices)
                del vali_hypothesis
                del vali_cost
                
                
        if avg_cost < 10:
            pltmode = 0
        if pltmode == 1:
            stack_loss = np.append(stack_loss,avg_cost.detach().to('cpu'))
            stack_vali_loss = np.append(stack_vali_loss, vali_avg_cost.detach().to('cpu'))
            stack_epoch = np.append(stack_epoch, epoch)
            plt.plot(stack_epoch, stack_loss)
            plt.plot(stack_epoch, stack_vali_loss)
            plt.show()
            
        print('{} [{:>4}] cost= [{:>.5} / {:>.5}] time= {:>.3}, ETA= {:>.3}h'.format(model_N,epoch + 1, avg_cost, vali_avg_cost,time.time()-pre_T, (time.time()-pre_T)*(training_epochs-epoch)/3600))
        pre_T = time.time()    
        if plot_mode == 1:
            train_imgs = []
            vali_imgs = []
            for i in range(len(train_plot_index)):
                plot_Y = train_balls_frame.iloc[int(train_plot_index[i]),:2]
                plot_Y[:2] = plot_Y[:2]/(720/img_height)
                train_X = cv2.imread("{}/{}".format(train_path,train_balls_frame.iloc[int(train_plot_index[i]),4]))
    #                train_X = cv2.cvtColor(train_X, cv2.COLOR_BGR2GRAY)
                train_X = cv2.resize(train_X,(img_width,img_height), interpolation=cv2.INTER_AREA)
                train_X = train_X.reshape(img_height,img_width,3)        
                train_X_tensor = torch.cuda.FloatTensor(train_X.transpose(2,0,1))
    #            train_X_tensor = torch.cuda.FloatTensor(train_X.transpose(2,0,1))
                train_Y = model(train_X_tensor.reshape([1,3,img_height,img_width]))
                train_Y = train_Y.detach().to('cpu')
                train_Y = train_Y.reshape([4])
                train_Y[0] = train_Y[0]*img_width
                train_Y[1] = train_Y[1]*img_height
    #            train_iou = iou_1(train_Y, plot_Y)
    #            cv2.putText(train_X, "IOU = {:>.4}".format(float(train_iou)), (1,5), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0,255),1)
                cv2.circle(train_X,(int(train_Y[0]),int(train_Y[1])),10,(255,255,0),-1)
                cv2.circle(train_X,(int(plot_Y[0]),int(plot_Y[1])),10,(0,0,255),-1)
    #            cv2.rectangle(train_X,
    #                          (int(train_Y[0]-0.5*train_Y[2]),int(train_Y[1]-0.5*train_Y[2])),
    #                          (int(train_Y[0]+0.5*train_Y[2]),int(train_Y[1]+0.5*train_Y[2])),(255,255,255),1)
    #            cv2.rectangle(train_X,
    #                          (int(plot_Y[0]-0.5*plot_Y[2]),int(plot_Y[1]-0.5*plot_Y[2])),
    #                          (int(plot_Y[0]+0.5*plot_Y[2]),int(plot_Y[1]+0.5*plot_Y[2])),(0,0,255),1)
                train_X  = cv2.resize(train_X,(int(0.4*img_width),int(0.4*img_height)), interpolation=cv2.INTER_AREA)
                if i == 0:
                    train_imgs = train_X
                else:
                    train_imgs = np.concatenate((train_imgs,train_X),axis=1)
            for i in range(len(vali_plot_index)):
                refer_Y_vali = train_balls_frame.iloc[int(vali_plot_index[i]),:2]
                refer_Y_vali[:2] = refer_Y_vali[:2]/(720/img_height)
                plot_vali_X = cv2.imread("{}/{}".format(train_path,train_balls_frame.iloc[int(vali_plot_index[i]),4]))
    #                plot_vali_X = cv2.cvtColor(plot_vali_X, cv2.COLOR_BGR2GRAY)
                plot_vali_X = cv2.resize(plot_vali_X,(img_width,img_height), interpolation=cv2.INTER_AREA)
                plot_vali_X = plot_vali_X.reshape(img_height,img_width, 3) 
                plot_vali_X_tensor = torch.cuda.FloatTensor(plot_vali_X.transpose(2,0,1))
                plot_vali_Y = model(plot_vali_X_tensor.reshape([1,3,img_height,img_width]))
                plot_vali_Y = plot_vali_Y.to('cpu')
                plot_vali_Y = plot_vali_Y.reshape([4])
                plot_vali_Y[0] = plot_vali_Y[0]*img_width
                plot_vali_Y[1] = plot_vali_Y[1]*img_height
    #            vali_iou = iou_1(plot_vali_Y, refer_Y_vali)
    #            cv2.putText(plot_vali_X, "IOU = {:>.4}".format(float(vali_iou)), (1,5), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0,255),1)
                cv2.circle(plot_vali_X,(int(plot_vali_Y[0]),int(plot_vali_Y[1])),10,(255,255,0),-1)
                cv2.circle(plot_vali_X,(int(refer_Y_vali[0]),int(refer_Y_vali[1])),10,(0,0,255),-1)
    #            cv2.rectangle(plot_vali_X,
    #                          (int(plot_vali_Y[0]-0.5*plot_vali_Y[2]),int(plot_vali_Y[1]-0.5*plot_vali_Y[2])),
    #                          (int(plot_vali_Y[0]+0.5*plot_vali_Y[2]),int(plot_vali_Y[1]+0.5*plot_vali_Y[2])),(255,255,255),1)
    #            cv2.rectangle(plot_vali_X,
    #                          (int(refer_Y_vali[0]-0.5*refer_Y_vali[2]),int(refer_Y_vali[1]-0.5*refer_Y_vali[2])),
    #                          (int(refer_Y_vali[0]+0.5*refer_Y_vali[2]),int(refer_Y_vali[1]+0.5*refer_Y_vali[2])),(0,0,255),1)
                plot_vali_X = cv2.resize(plot_vali_X,(int(0.4*img_width),int(0.4*img_height)), interpolation=cv2.INTER_AREA)
                if i == 0:
                    vali_imgs = plot_vali_X
                else:
                    vali_imgs = np.concatenate((vali_imgs,plot_vali_X),axis=1)
            total_imgs = np.concatenate((train_imgs,vali_imgs),axis=0)
            total_imgs = cv2.resize(total_imgs,(1280,720), interpolation=cv2.INTER_AREA)
            
  #          cv2.imshow('{}'.format(model_N),total_imgs)
  #          cv2.waitKey(10)           
            cv2.imwrite('{}/{}_{}_{}train.jpg'.format(trainpath,i, epoch +1, avg_cost),total_imgs)

         ## save model
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
