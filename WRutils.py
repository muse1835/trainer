from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
from skimage import io, transform
import cv2
class Anafidataset(Dataset):
            
        def __init__(self, root_dir):
            """
            Args:
                csv_file (string): csv 파일의 경로
                root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
                transform (callable, optional): 샘플에 적용될 Optional transform
            """
            self.label_list = os.listdir('{}/labels'.format(root_dir))
            self.root_dir = root_dir
            self.transform = 'transform'
            
        def __len__(self):
            return len(self.label_list)
    
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            name = os.path.splitext(self.label_list[idx])[0]
            image = io.imread('{}/images/{}.jpg'.format(self.root_dir,name))
            if image.shape[1] != 100:
                image = cv2.resize(image,(100,100), interpolation=cv2.INTER_AREA)
            image = image.transpose(2,0,1)
            label = list(np.loadtxt('{}/labels/{}.txt'.format(self.root_dir, name)))
            label[:4] = [label[i]/100 for i in range(4)]

            return torch.FloatTensor(image), torch.FloatTensor(label)
        

class Anafi_model(torch.nn.Module):
    def __init__(self):
        super(Anafi_model, self).__init__()
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
            torch.nn.Conv2d(256, 5, kernel_size=3, stride=3, padding=0))


    def forward(self, x):
        out = self.layer1(x)
        out = out.view(-1, 5)   # Flatten them for FC
        return out
