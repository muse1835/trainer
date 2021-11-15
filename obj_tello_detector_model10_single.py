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
pltmode = 0
stack_loss = []
stack_vali_loss = []
stack_epoch = []
pre_T = 0
plot_mode = 1

with torch.cuda.device(0):    

    for seed in range(778,779):  
        
        model_N = '10'
        folder = '210701_crop_input100__model{}_856k'.format(model_N)
        # path = "tello_label_no rot"
        path = "total_0417/crop"
        train_path = "total_0417/crop"
        save_path = '0701_parameter_diet'
        def iou_(input, target):
            sum_iou = 0
            for i in range(len(Y)):
                boxA = 400*[input[i,0]-0.5*input[i,2],  input[i,1]-0.5*input[i,3],  input[i,0]+0.5*input[i,2],  input[i,1]+0.5*input[i,3]]
                boxB = 400*[target[i,0]-0.5*target[i,2],target[i,1]-0.5*target[i,3],target[i,0]+0.5*target[i,2],target[i,1]+0.5*target[i,3]]
            	# determine the (x, y)-coordinates of the intersection rectangle
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
            	# compute the area of intersection rectangle
                interArea = abs(max(0, xB - xA) * max(0, yB - yA ))
                if interArea == 0:
                    return 0
            	# compute the area of both the prediction and ground-truth
            	# rectangles
                boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
                boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
            	# compute the intersection over union by taking the intersection
            	# area and dividing it by the sum of prediction + ground-truth
            	# areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
            	# return the intersection over union value
                sum_iou = sum_iou + iou    
            return sum_iou
        
        def iou_1(input, target):
            # input = input.detach().cpu().numpy()
            # target = target.detach().cpu().numpy()
            #Box point
            boxA = 400*[input[0]-0.5*input[2],  input[1]-0.5*input[3],  input[0]+0.5*input[2],  input[1]+0.5*input[3]]
            boxB = 400*[target[0]-0.5*target[2],target[1]-0.5*target[3],target[0]+0.5*target[2],target[1]+0.5*target[3]]
        	# determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
        	# compute the area of intersection rectangle
            interArea = abs(max(0, xB - xA) * max(0, yB - yA ))
            if interArea == 0:
                return 0
        	# compute the area of both the prediction and ground-truth
        	# rectangles
            boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
            boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        	# compute the intersection over union by taking the intersection
        	# area and dividing it by the sum of prediction + ground-truth
        	# areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
        	# return the intersection over union value
            return iou
        def SSE_loss(input, target):
            sum_loss = 0
            for i in range(len(input)):
                if sum(target[i,:2]) != 0:
                # loss = (input[i,0]-target[i,0])**2+ (input[i,1]-target[i,1])**2+ (torch.sign(input[i,2])*torch.sqrt(torch.abs(input[i,2]))-torch.sign(target[i,2])*torch.sqrt(torch.abs(target[i,2])))**2
                    loss = (input[i,0]-target[i,0])**2+ (input[i,1]-target[i,1])**2+ (input[i,2]-target[i,2])**2 + (input[i,3]-target[i,3])**2
                else:
                    loss = (input[i,3]-target[i,3])**2
                sum_loss = sum_loss + loss
            return sum_loss
        
        def VSSE_loss(input, target):
            loss = 0
            # loss = (input[0]-target[0])**2+ (input[1]-target[1])**2+ (torch.sign(input[2])*torch.sqrt(torch.abs(input[2]))-torch.sign(target[2])*torch.sqrt(torch.abs(target[2])))**2
            loss = (input[0]-target[0])**2+ (input[1]-target[1])**2+ (input[2]-target[2])**2 + (input[i,3]-target[i,3])**2
            return loss
        
        
        def objectness_loss(input, target):
            sum_loss = 0
            for i in range(len(input)):
                sum_loss += (input[i,3]-target[i,3])**2
            return sum_loss
            
        # time : 431.08308506011963
        start = time.time()  # 시작 시간 저장
        
        # plt.ion()  
        # device = torch.device("cuda")
        
        # for reproducibility
        torch.cuda.manual_seed(seed)
        # if device == 'cuda':
        #     torch.cuda.manual_seed_all(777)
            
            
            # parameters
        image_size = 100
        learning_rate = 0.0001
        training_epochs = 200000
        batch_size = 128
        
        ## image dataset
        train_balls_frame = pd.read_csv('{}/data.csv'.format(train_path))
    #    negative_frame = pd.read_csv('{}/data.csv'.format(negative_path))
        n = len(train_balls_frame)
    
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
        class Balldataset(Dataset):
            """Face Landmarks dataset."""
        
            def __init__(self, csv_file, root_dir, transform=None):
                """
                Args:
                    csv_file (string): csv 파일의 경로
                    root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
                    transform (callable, optional): 샘플에 적용될 Optional transform
                """
                self.balls_frame = pd.read_csv('{}/data.csv'.format(root_dir))
                self.root_dir = root_dir
                self.transform = 'transform'
                
            def __len__(self):
                # print(len(self.balls_frame))
                return len(self.balls_frame)
        
            def __getitem__(self, idx):
                if torch.is_tensor(idx):
                    idx = idx.tolist()
        
                img_name = os.path.join(self.root_dir,
                                        self.balls_frame.iloc[idx, 4])
                image = io.imread(img_name)
                image = cv2.resize(image,(image_size,image_size), interpolation=cv2.INTER_AREA)
                image = image.reshape(image_size, image_size, 3)
                image = image.transpose(2,0,1)
                # image = image[0, :, :]
                # image = image.reshape(1,image_size,image_size)
                # image = image.reshape(1,image_size,image_size)
                # image[0,:,:] = (image[0,:,:] - np.min(image[0,:,:]))/ (np.max(image[0,:,:]) - np.min(image[0,:,:]))
                # image[1,:,:] = (image[1,:,:] - np.min(image[1,:,:]))/ (np.max(image[1,:,:]) - np.min(image[1,:,:]))
                # image[2,:,:] = (image[2,:,:] - np.min(image[2,:,:]))/ (np.max(image[2,:,:]) - np.min(image[2,:,:]))
                # image[0,:,:] = (image[0,:,:] - np.mean(image[0,:,:]))/ np.std(image[0,:,:])
                # image[1,:,:] = (image[1,:,:] - np.mean(image[1,:,:]))/ np.std(image[1,:,:])
                # image[2,:,:] = (image[2,:,:] - np.mean(image[2,:,:]))/ np.std(image[2,:,:])
                balls = self.balls_frame.iloc[idx, :4]
                balls[:4] = balls[:4]/200
                if sum(balls[:2]) != 0:
                    balls[:2] = balls[:2]
                # balls[3] = balls[3] * 50
                
                balls = np.array([balls])
                balls = balls.astype('float').reshape(4)
                # sample = {'image': image, 'balls': balls}
                # X = torch.cuda.FloatTensor(image)
                # Y = torch.cuda.FloatTensor(balls)
                X = torch.tensor(image).float()
                Y = torch.tensor(balls).float()
                # Y = torch.tensor(balls, dtype=torch.long, device=device)
                # if self.transform:
                #     sample = self.transform(sample)
        
                # return sample
                return X, Y
            
            
        train_set = Balldataset(csv_file='{}/data.csv'.format(train_path),
                                        root_dir=train_path)
    #    negative_set = Balldataset(csv_file='{}/data.csv'.format(negative_path),
    #                                    root_dir=negative_path)
        
        train, vali = torch.utils.data.random_split(train_set, [num_train, num_vali])
    #    negative, _ = torch.utils.data.random_split(negative_set, [int(0.1*len(negative_frame)),negative_dummy ])
        
        
        train_dataset = torch.utils.data.DataLoader(dataset=train,
                                                   batch_size=batch_size,
                                                  # num_workers=4, 
                                                  shuffle=True,
                                                  drop_last=False,
                                                  pin_memory = True)
        vali_dataset = torch.utils.data.DataLoader(dataset=vali,
                                                   batch_size=batch_size,
                                                  # num_workers=4, 
                                                  shuffle=True,
                                                  drop_last=False,
                                                  pin_memory = True)
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
        
        # image = Variable(image)
        
        # ball_train = image
        
        # mnist_train = dsets.MNIST(root='MNIST_data/',
        #                           train=True,
        #                           transform=transforms.ToTensor(),
        #                           download=True)
        
        # mnist_test = dsets.MNIST(root='MNIST_data/',
        #                           train=False,
        #                           transform=transforms.ToTensor(),
        #                           download=True)
        
        
        
        # data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
        #                                           batch_size=batch_size,
        #                                           shuffle=True,
        #                                           drop_last=False)
        
    
        # CNN Model (2 conv layers)
        class ball_detect(torch.nn.Module):
        
            def __init__(self):
                super(ball_detect, self).__init__()
                # L1 ImgIn shape=(?, 28, 28, 1)
                #    Conv     -> (?, 28, 28, 32)
                #    Pool     -> (?, 14, 14, 32)
                self.layer1 = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(3),
                    torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
                    
                    torch.nn.BatchNorm2d(16),
                    torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
    
                    torch.nn.BatchNorm2d(32),
                    torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
    
                    torch.nn.BatchNorm2d(16),                    
                    torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
        
                    torch.nn.BatchNorm2d(32),
                    torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
        
                    torch.nn.BatchNorm2d(16),
                    torch.nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
    
                    torch.nn.BatchNorm2d(8),
                    torch.nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
                    
                    torch.nn.BatchNorm2d(32),
                    torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
    
                    torch.nn.BatchNorm2d(16),
                    torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
    
                    torch.nn.BatchNorm2d(32),
                    torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
                    
                    torch.nn.BatchNorm2d(64),
                    torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
                    
                    torch.nn.BatchNorm2d(128),
                    torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
                    
                    torch.nn.BatchNorm2d(256),
                    torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
                    
                    torch.nn.BatchNorm2d(128),
                    torch.nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Dropout(0.2),
        #                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #                torch.nn.LeakyReLU(),
        #                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #                torch.nn.BatchNorm2d(64),
        #                torch.nn.Dropout(0.2),
        #                
        #                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #                torch.nn.LeakyReLU(),
        #                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #                torch.nn.BatchNorm2d(128),
        #                torch.nn.Dropout(0.2),
        #                
        #                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #                torch.nn.LeakyReLU(),
        #                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #                torch.nn.BatchNorm2d(64),
        #                torch.nn.Dropout(0.2),
        
                    
                    # torch.nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                    # torch.nn.LeakyReLU(),
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    # torch.nn.BatchNorm2d(512),
                    # torch.nn.Dropout(0.2),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.Conv2d(256, 4, kernel_size=3, stride=3, padding=0)).cuda()
        
        
                # Final FC 7x7x64 inputs -> 10 outputs
                # self.fc = torch.nn.Linear(75, 3, bias=True).cuda()
                # torch.nn.init.xavier_uniform_(self.fc.weight).cuda()
        
            def forward(self, x):
                out = self.layer1(x.cuda()).cuda()
                # out = torch.mean(x.view(x.size(0), x.size(1), 1,1),dim=2)
                out = out.view(-1, 4).cuda()   # Flatten them for FC
                # out = self.fc(out)
                return out
            
    
    
            
            
        # instantiate CNN model
        # model = nn.DataParallel(CNN().cuda(), device_ids=[2,3])
        # CNN().cuda()
        model = ball_detect().cuda()
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
        
        
        
        
        #create new folder
        modelpath = "{}//model{}".format(save_path,folder)
        trainpath = "{}//train{}".format(save_path,folder)
        valipath = "{}//vali{}".format(save_path,folder)
        if not os.path.exists(modelpath):
            os.makedirs(modelpath)
            os.makedirs(trainpath)
            os.makedirs(valipath)
        
        
        model.train()
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
            for X, Y in train_dataset:
                # X = torch.cuda.FloatTensor(X)
                # Y = torch.cuda.FloatTensor(Y)
                # print(len(X))
                optimizer.zero_grad()
                hypothesis = model(X)
                YY = hypothesis.detach().to('cpu')
                cost = SSE_loss(hypothesis, Y)
                cost.backward()
                optimizer.step()
                Y = Y.detach().to('cpu')
                if Y[0][2] != 0:
                    train_IOU = iou_(YY, Y)
                    train_count += 1
                train_mIOU += train_IOU / len(train_dataset.dataset.indices)
                avg_cost += cost / len(train_dataset.dataset.indices)
                # print("1")
                
                del hypothesis
                del cost
    #        for X, Y in negative_dataset:
    #            # X = X.reshape(-1,1,400,400)
    #            # Y = Y.cuda()
    #            # print(len(X))
    #            optimizer.zero_grad()
    #            hypothesis = model(X)
    #            YY = hypothesis.detach().to('cpu')
    #            cost = objectness_loss(hypothesis, Y)
    #            cost.backward()
    #            optimizer.step()
    #            Y = Y.detach().to('cpu')
    #            avg_cost += cost / len(negative_dataset.dataset.indices)
    #            # print("1")
    #            
    #            del hypothesis
    #            del cost
        
        
            for X, Y in vali_dataset:
                with torch.no_grad():
                    # vali_X = X.reshape(-1,1,400,400)
                    # vali_Y = Y.cuda()
                    # vali_X = torch.cuda.FloatTensor(X)
                    # vali_Y = torch.cuda.FloatTensor(Y)
                    vali_X = X
                    vali_Y = Y
                    vali_hypothesis = model(vali_X)
                    vali_YY = vali_hypothesis.detach().to('cpu')
                    vali_cost = SSE_loss(vali_hypothesis, vali_Y)
                    if Y[0][2] != 0:
                        vali_IOU = iou_(vali_YY, vali_Y.detach().to('cpu'))
                        vali_count += 1
                    vali_mIOU += vali_IOU / len(vali_dataset.dataset.indices)
                    vali_avg_cost += vali_cost / len(vali_dataset.dataset.indices)
                    # for i in range(len(Y)):
                    #     vali_cost = VSSE_loss(hypothesis[i,:], vali_Y[i,:])
                    #     vali_avg_cost += vali_cost / len(vali_data.dataset.indices)
            if avg_cost < 0.4:
                pltmode = 1
            if pltmode == 1:
                stack_loss = np.append(stack_loss,avg_cost.detach().to('cpu'))
                stack_vali_loss = np.append(stack_vali_loss, vali_avg_cost.detach().to('cpu'))
                stack_epoch = np.append(stack_epoch, epoch)
                plt.plot(stack_epoch, stack_loss)
                plt.plot(stack_epoch, stack_vali_loss)
                plt.show()
    
    #        print('[Epoch: {:>4}] cost = {:>.9} vali_cost = {:>.9}'.format(epoch + 1, avg_cost, vali_avg_cost))
            print('{} [Epoch: {:>4}] cost = {:>.9} vali_cost = {:>.9} train_mIOU = {:>.9} vali_mIOU = {:>.9} time = {:>.5}'.format(model_N,epoch + 1, avg_cost, vali_avg_cost, train_mIOU, vali_mIOU,time.time()-pre_T))
            pre_T = time.time()
    
            ## plot refernce / train / vali  
            # plot_X = cv2.imread("label_no_aug/{}.jpg".format(train_plot_index))
            # plot_Y = balls_frame.iloc[train_plot_index-1,:3]
            # plot_Y[:2] = plot_Y[:2]+200
        
            # cv2.circle(plot_X,(int(plot_Y[0]),int(plot_Y[1])),10,(0,255,255))    # X_np = X.detach().cpu().numpy()
            # cv2.rectangle(plot_X,(int(plot_Y[0]-0.5*plot_Y[2]),int(plot_Y[1]-0.5*plot_Y[2])),(int(plot_Y[0]+0.5*plot_Y[2]),int(plot_Y[1]+0.5*plot_Y[2])),10)
            # cv2.imshow('X_data',plot_X)
            # cv2.imwrite('label_no_aug/4refer/{}_{}label.jpg'.format(epoch+1, avg_cost),plot_X)
        
        
        ##1008 del
            if plot_mode == 1:
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
                    train_Y = train_Y.reshape([4])
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
                    cv2.imshow('train_data_{}_{}'.format(i,model_N),train_X)
                    cv2.imwrite('{}/result/train{}/{}_{}_{}train.jpg'.format(path,folder,i, epoch +1, avg_cost),train_X)
                for i in range(len(vali_plot_index)):
                    refer_Y_vali = train_balls_frame.iloc[int(vali_plot_index[i]),:4]
                    # if sum(refer_Y_vali[:2]) != 0:
                    refer_Y_vali[:4] = refer_Y_vali[:4]/(200/image_size)
                    plot_vali_X = cv2.imread("{}/{}".format(train_path,train_balls_frame.iloc[int(vali_plot_index[i]),4]))
                    plot_vali_X = cv2.resize(plot_vali_X,(image_size,image_size), interpolation=cv2.INTER_AREA)
                    plot_vali_X_tensor = torch.cuda.FloatTensor(plot_vali_X.transpose(2,0,1))
                    plot_vali_Y = model(plot_vali_X_tensor.reshape([1,3,image_size,image_size]))
                    plot_vali_Y = plot_vali_Y.to('cpu')
                    plot_vali_Y = plot_vali_Y.reshape([4])
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
                    cv2.imshow('vali_data_{}_{}'.format(i,model_N),plot_vali_X)
                    cv2.imwrite('{}/result/vali{}/{}_{}_{}vali.jpg'.format(path,folder,i,epoch+1,vali_avg_cost),plot_vali_X)
                cv2.waitKey(10)
            
            
            
    #        result save
    #        for i in range(len(train_dataset.dataset.indices)):
    #            plot_Y = train_balls_frame.iloc[int(train_dataset.dataset.indices[i]),:3]
    #            plot_Y[:2] = plot_Y[:2]
    #            train_X = cv2.imread("{}/{}.jpg".format(train_path,int(train_dataset.dataset.indices[i]+1)))
    #            train_X_tensor = torch.cuda.FloatTensor(train_X.transpose(2,0,1))
    ##            train_X_tensor = torch.cuda.FloatTensor(train_X.transpose(2,0,1))
    #            train_Y = model(train_X_tensor.reshape([1,3,400,400]))
    #            train_Y = train_Y.detach().to('cpu')
    #            train_Y = train_Y.reshape([3])
    #            train_Y[:2] = train_Y[:2] +200
    #            train_iou = iou_1(train_Y, plot_Y)
    #            cv2.putText(train_X, "IOU = {:>.4}".format(float(train_iou)), (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0,255),3)
    #            cv2.circle(train_X,(int(train_Y[0]),int(train_Y[1])),10,(255,255,255))
    #            cv2.circle(train_X,(int(plot_Y[0]),int(plot_Y[1])),10,(0,255,255))
    #            cv2.rectangle(train_X,
    #                          (int(train_Y[0]-0.5*train_Y[2]),int(train_Y[1]-0.5*train_Y[2])),
    #                          (int(train_Y[0]+0.5*train_Y[2]),int(train_Y[1]+0.5*train_Y[2])),(255,255,255),3)
    #            cv2.rectangle(train_X,
    #                          (int(plot_Y[0]-0.5*plot_Y[2]),int(plot_Y[1]-0.5*plot_Y[2])),
    #                          (int(plot_Y[0]+0.5*plot_Y[2]),int(plot_Y[1]+0.5*plot_Y[2])),(0,0,255),3)
    #            train_X  = cv2.resize(train_X,(300,300), interpolation=cv2.INTER_AREA)
    ##            cv2.imshow('train_data_{}'.format(i),train_X)
    #            cv2.imwrite('{}/result/train_lst1/{}_{}_{}train.jpg'.format(path,i, epoch +1, avg_cost),train_X)
    #    
    #        for i in range(len(vali_dataset.dataset.indices)):
    #            refer_Y_vali = train_balls_frame.iloc[int(vali_dataset.dataset.indices[i]),:3]
    #            refer_Y_vali[:2] = refer_Y_vali[:2]
    #            plot_vali_X = cv2.imread("{}/{}.jpg".format(train_path,int(vali_dataset.dataset.indices[i]+1)))
    #            plot_vali_X_tensor = torch.cuda.FloatTensor(plot_vali_X.transpose(2,0,1))
    #            plot_vali_Y = model(plot_vali_X_tensor.reshape([1,3,400,400]))
    #            plot_vali_Y = plot_vali_Y.to('cpu')
    #            plot_vali_Y = plot_vali_Y.reshape([3])
    #            plot_vali_Y[:2] = plot_vali_Y[:2] +200
    #            vali_iou = iou_1(plot_vali_Y, refer_Y_vali)
    #            cv2.putText(plot_vali_X, "IOU = {:>.4}".format(float(vali_iou)), (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0,255),3)
    #            cv2.circle(plot_vali_X,(int(plot_vali_Y[0]),int(plot_vali_Y[1])),10,(255,255,255))
    #            cv2.circle(plot_vali_X,(int(refer_Y_vali[0]),int(refer_Y_vali[1])),10,(0,255,255))
    #            cv2.rectangle(plot_vali_X,
    #                          (int(plot_vali_Y[0]-0.5*plot_vali_Y[2]),int(plot_vali_Y[1]-0.5*plot_vali_Y[2])),
    #                          (int(plot_vali_Y[0]+0.5*plot_vali_Y[2]),int(plot_vali_Y[1]+0.5*plot_vali_Y[2])),(255,255,255),3)
    #            cv2.rectangle(plot_vali_X,
    #                          (int(refer_Y_vali[0]-0.5*refer_Y_vali[2]),int(refer_Y_vali[1]-0.5*refer_Y_vali[2])),
    #                          (int(refer_Y_vali[0]+0.5*refer_Y_vali[2]),int(refer_Y_vali[1]+0.5*refer_Y_vali[2])),(0,0,255),3)
    #            plot_vali_X = cv2.resize(plot_vali_X,(300,300), interpolation=cv2.INTER_AREA)
    ##            cv2.imshow('vali_data_{}'.format(i),plot_vali_X)
    #            cv2.imwrite('{}/result/tt/{}_{}_{}vali.jpg'.format(path,vali_dataset.dataset.indices[i],epoch+1,vali_avg_cost),plot_vali_X)
        
        
        
        
        
        
            # print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
            if epoch%10 == 0:        
                torch.save(model.state_dict(), "{}/model{}/cost{}_{}.pth".format(save_path,folder,epoch+1,avg_cost))
        print('Learning Finished!')
        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    