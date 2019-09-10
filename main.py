#!/usr/bin/env python
# coding: utf-8

# In[65]:


from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from math import sqrt
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler


trainImgPath = "/home/lewis/Desktop/1project/Desktop/object_segmentation/severstal-steel-defect-detection/train_images/"
trainCsv = "/home/lewis/Desktop/1project/Desktop/object_segmentation/severstal-steel-defect-detection/train.csv"
df = pd.read_csv(trainCsv)
#df = df[~df['EncodedPixels'].isnull()]# get only image with labeled data for defects
#print(df['EncodedPixels'])

#print(df['ImageId_ClassId'])
Imid=[x.split('_')[0] for x in df['ImageId_ClassId'] ]
print(len(Imid)//4)
#print(Imid)
print(df.shape)

from skimage.io import imread
from scipy.ndimage.filters import convolve


# In[2]:


import cv2 
import numpy as np 
from skimage.io import imread
def drawContours(image):
    edged = cv2.Canny(image, 230, 240) 
    print(edged.shape)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    cv2.drawContours(image, contours, -1, (0, 255, 0),1) 
    return image
plt.figure(figsize=(35,10))
plt.imshow(drawContours(cv2.imread(trainImgPath+"0002cc93b.jpg")), cmap = 'Greys', interpolation = 'bicubic')


# In[3]:


plt.figure(figsize=(35,10))
plt.imshow(cv2.imread(trainImgPath+"0002cc93b.jpg"), cmap = 'Greys', interpolation = 'bicubic')


# In[14]:

#class_number stand for this time we take which class of mask
def getMaskByClass(listEncodedString, listLabels, class_number=1):
    mask = np.zeros((256, 1600, 4), dtype=np.float64)
    if len(str(listEncodedString))==0:
        return mask
    for encodedString,labels in zip (listEncodedString, listLabels):
        encodedString = str(encodedString).split(" ")
        flatmask = np.zeros(1600*256)
        for i in range(0,len(encodedString)//2):
            start = int(encodedString[2*i])
            end = int(encodedString[2*i]) +int(encodedString[2*i+1])
            flatmask[start:end-1] =  3
        mask[:,:,labels-1] = np.transpose(flatmask.reshape(1600,256))
    return mask


# In[61]:


"""
out=getMaskByClass(df['EncodedPixels'][idx*4:idx*4+4],[1,2,3,4])
    print(out.shape)

    plt.figure(figsize=(35,10))
    plt.imshow(cv2.imread(trainImgPath+Imid[idx*4]), cmap = 'Greys', interpolation = 'bicubic')
    plt.figure(figsize=(35,10))
    plt.imshow(out[:,:,0], cmap = 'Greys', interpolation = 'bicubic')
    plt.figure(figsize=(35,10))
    plt.imshow(out[:,:,1], cmap = 'Greys', interpolation = 'bicubic')
    plt.figure(figsize=(35,10))
    plt.imshow(out[:,:,2], cmap = 'Greys', interpolation = 'bicubic')
    plt.figure(figsize=(35,10))
    plt.imshow(out[:,:,3], cmap = 'Greys', interpolation = 'bicubic')"""

# to select which class to be used, see here
def output_data(idx):
    return {'image':cv2.imread(trainImgPath+Imid[idx*4]),'masks':getMaskByClass(df['EncodedPixels'][idx*4:idx*4+4],[1,2,3,4])}
print(output_data(0)['image'].shape)
i=0

plt.figure(figsize=(35,10))
plt.imshow(output_data(0)['image'], cmap = 'Greys', interpolation = 'bicubic')
plt.figure(figsize=(35,10))
plt.imshow(output_data(0)['masks'][:,:,0], cmap = 'Greys', interpolation = 'bicubic')


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = np.array(y_true).flatten()
    y_pred_f = np.array(y_pred).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = np.array(y_true).flatten()
    y_pred_f = np.array(y_pred).flatten()
    intersection = y_true_f * y_pred_f
    score = (2. * np.sum(intersection) + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


#using pytorch to train the unet
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

       
        return {'image': torch.from_numpy(image),
                'masks': torch.from_numpy(masks)}
    
class data_loader(Dataset):
    def __init__(self, class_number, transform=None):
        self.transform = transform
        self.class_number=class_number
    def __len__(self):
        return len(Imid)//4
    def __getitem__(self, idx):
        sample=output_data(idx)
        #print(sample['masks'].shape)
        single_mask=sample['masks'][:,:,self.class_number-1]
        if self.transform:
            sample = self.transform(sample)
        return {'image':sample['image'], 'masks':single_mask}


#sellect the class number to train here

face_dataset = data_loader(1,transform=transforms.Compose([ToTensor()]))
#print('mask_transform',face_dataset[0]['masks'][:,:,0:1].shape)


validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(face_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
batch_size=3
train_loader = torch.utils.data.DataLoader(face_dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(face_dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

# Usag

import torch
from torchvision import transforms, datasets



#face_dataloader = torch.utils.data.DataLoader(face_dataset, batch_size= 3, shuffle= False, num_workers= 2)




"""
for i, data in enumerate(face_dataloader, 0):
    print('mask',data['masks'][0])
    for i in range(data['masks'][0].shape[0]):
        for j in range(data['masks'][0].shape[1]):
            if data['masks'][0][i][j]!=0:
                print(data['masks'][0][i][j])
"""

    # PIL
    #img = transforms.ToPILImage()(data[i][0])
    #img.show()
    #break


# In[100]:


import torch
import torch.nn as nn
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self,colordim =3):
        super(UNet, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.layer1 = double_conv(3,64)
        self.maxpool1to2=nn.MaxPool2d(2)

        self.layer2 = double_conv(64,128)
        self.maxpool2to3=nn.MaxPool2d(2)

        self.layer3 = double_conv(128, 256)
        self.maxpool3to4 = nn.MaxPool2d(2)

        self.layer4 = double_conv(256, 512)
        self.maxpool4to5 = nn.MaxPool2d(2)

        self.layer5 = double_conv(512, 1024)
        #self.maxpool5to6 = nn.MaxPool2d(2)

        #self.layer6=
        #Upsampling
        self.up1=nn.ConvTranspose2d(1024, 512, kernel_size =(2, 2), stride =2)


        self.layer6 = double_conv(1024, 512)
        self.up2=nn.ConvTranspose2d(512, 256, kernel_size =(2, 2), stride =2)

        self.layer7 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=2)

        self.layer8 = double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=2)

        self.layer9 = double_conv(128, 64)
        self.final=nn.Conv2d(64,1,1)
    def forward(self, x1):
        #print(x1.shape)
        x2=self.layer1(x1)
        x3=self.maxpool1to2(x2)
        #print(x3.shape)
        x4 = self.layer2(x3)
        x5 = self.maxpool2to3(x4)
        #print(x5.shape)
        x6 = self.layer3(x5)
        x7 = self.maxpool3to4(x6)
        #print('x7',x7.shape)
        x8 = self.layer4(x7)
        #print('x8',x8.shape)
        x9 = self.maxpool4to5(x8)
        #print(x9.shape)

        x10 = self.layer5(x9)
        #x11 = self.maxpool5to6(x10)
        x11=self.up1(x10)
        #print('x11',x11.shape)
        x12=torch.cat((x11,x8),1)
        #print('x12',x12.shape)

        x13=self.layer6(x12)
        x14 = self.up2(x13)
        x15 = torch.cat((x14, x6), 1)
        #print('x15',x15.shape)

        x16=self.layer7(x15)
        x17 = self.up3(x16)
        x18 = torch.cat((x17, x4), 1)
        #print('x18',x18.shape)

        x19 = self.layer8(x18)
        x20 = self.up4(x19)
        x21 = torch.cat((x20, x2), 1)
        #print('x21',x21.shape)

        x22=self.layer9(x21)
        #print('x22',x22.shape)

        #put a sigmoid here
        output=self.final(x22)
        output=nn.functional.sigmoid(output)
        return output



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


unet = UNet(3).cuda()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet=unet.to(device)


#start training the model

import torch.optim as optim

optimizer = optim.SGD(unet.parameters(), lr=0.00001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    running_loss=0.0
    for i, data in enumerate(train_loader, 0):
        #print(data['image'][0].shape)
        # PIL
        #img = transforms.ToPILImage()(data[i][0])
        #img.show()
        #print('image shape',data['image'].shape)
        input=data['image'].permute(0,3,1,2)
        #print('masks shape',data['masks'].shape)
        input=input.float().cuda()
        #print(unet(input).shape)
        #print('generated_mask',generated_mask.shape)

        true_mask=data['masks'].cuda()
        #print('true_mask',true_mask.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        generated_mask=unet(input).double()
        generated_mask=generated_mask.view(-1,256,1600)
        #print(true_mask)
        criteria=torch.nn.BCELoss()
        assert ((true_mask/3 >= 0.) & (true_mask/3 <= 1.)).all()
        assert ((generated_mask / 3 >= 0.) & (generated_mask / 3 <= 1.)).all()
        print(generated_mask.shape)
        print(true_mask.shape)
        l2_loss=criteria(generated_mask,true_mask/3)
        #l2_loss = torch.sum(torch.abs(generated_mask - true_mask))
        print(l2_loss)
        #print(l2_loss)
        l2_loss.backward()
        optimizer.step()

        # print statistics
        running_loss += l2_loss.item()
        if i % 10 == 9:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0


#defind optimizer
torch.save(unet, './' + 'unet_batch3' + '.pt')


"""
unet.load_state_dict(torch.load('./_' + 'unet_batch3' + '.pt'))
unet.eval()
"""
