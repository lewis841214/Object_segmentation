#!/usr/bin/env python
# coding: utf-8

# In[65]:


from __future__ import print_function, division
import torch.backends.cudnn as cudnn
import time
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
            flatmask[start:end-1] =  1.0
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
        #single_mask=sample['masks'][:,:,self.class_number-1]
        if self.transform:
            sample = self.transform(sample)
        return sample


#sellect the class number to train here
#TODO: change to 4 dataset
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

    #todo change it back to normal set up
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)



#train_loader = torch.utils.data.DataLoader(face_dataset, batch_size=batch_size,
                                           #sampler=train_sampler)
#validation_loader = torch.utils.data.DataLoader(face_dataset, batch_size=batch_size,
                                            #  sampler=valid_sampler)
#data_loaders={"train":train_loader, "validate":validation_loader}


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
        self.final=nn.Conv2d(64,4,1)
    def forward(self, x1):
        #print('x1',x1.shape)
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
        #output=nn.functional.sigmoid(output)
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



#start training the model

import torch.optim as optim




#utility
def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    #print(pred.shape)
    #pred.astype(np.float16)
    label.astype(np.uint8)
    """
    #print(pred)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            for k in range(pred.shape[2]):
                if pred[i][j][k]!=0:
                    print('non zero',pred[i][j][k])
       
        for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            for k in range(label.shape[2]):
                if label[i][j][k]!=0:
                    print('non zero',label[i][j][k])
    """


    #print(label)
    for c in classes:
        #print('c',c)
        label_c = label == c
        #print('label_c',label_c.shape)
        #print( 'label_c', label_c)
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        # check the value of non zero elements in masks
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            #print(intersection)
            #print(union)
            #print(intersection / union)
            ious.append(intersection / union)
    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    #print(outputs.shape)
    #print(outputs)
    #print(labels.shape)
    labels=labels.permute(0,3,1,2)
    preds = np.copy(outputs) # copy is imp


    labels = np.array(labels) # tensor to np
    #print(preds.shape)
    #print(labels.shape)
    #print()

    for pred, label in zip(preds, labels):
        #print(pred.shape)
        #print(type(pred))
        #pred=pred.reshape(256,1600)
        #print(label.shape)
        #print(pred.shape)
        #print(label)
        #print(compute_ious(pred, label, classes))
        #print()
        """print(pred.shape)
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                for k in range(pred.shape[2]):
                    if pred[i][j][k] != 0:

                        a=0
                        print('non zero', pred[i][j][k])
        print('if end here')"""


        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou

class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model, load_dict=True):

        self.num_workers = 6
        self.batch_size = {"train": 2, "val": 2}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-2
        self.num_epochs = 20
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.load_dict = load_dict
        if self.load_dict==True:
            self.net.load_state_dict(torch.load("./model.pth")["state_dict"])
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)

        cudnn.benchmark = True
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(face_dataset, batch_size=self.batch_size['train'],
                                           sampler=train_sampler,drop_last=True),
            "val":torch.utils.data.DataLoader(face_dataset, batch_size=self.batch_size['val'],
                                           sampler=valid_sampler,drop_last=True),

        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        masks=masks.permute(0, 3, 1, 2)
        #print('image',images.shape)
        images = images.permute(0, 3, 1, 2)
        images=images.float().cuda()
        outputs = self.net(images)
        #print('outputs',outputs.shape)
        outputs=outputs.reshape(-1,4,256,1600)
        #print(outputs.shape)
        #print('masks',masks.shape)
        #print(type(outputs))
        #print(type(masks))
        masks=masks.float()
        loss = self.criterion(outputs, masks)
        #print(loss)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        #         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):  # replace `dataloader` with `tk0` for tqdm
            images = batch['image']
            targets=batch['masks']
            #print(images.shape)
            #print(targets.shape)
            #print(batch['image'])
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            print()
unet = UNet(3)
model_trainer = Trainer(unet)
model_trainer.start()
