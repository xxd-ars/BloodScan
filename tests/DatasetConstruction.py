#%%
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import random
from random import randrange
import time
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import glob
from skimage.util import montage
import cv2
from scipy import signal
from collections import Counter
from tqdm import tqdm
np.random.seed(0)

#%%
train_data_path = 'data_V1'
train_image_paths = []
validation_image_paths = []
classes = []

for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split(os.sep)[-1])
    for file in glob.glob(data_path+'/*.png'):
        if file.split('_')[-1]=='T3.png':
            train_image_paths.append(file)

print(len(train_image_paths))
print(classes)

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
print(idx_to_class)
print(class_to_idx)

#%%
def LoadData(img_paths,img_size,class_to_idx):
  n = len(img_paths)
  Images = np.zeros((n,img_size,img_size,3),dtype='uint8')
  Labels = np.zeros(n)
  for i in range(n):
    path = img_paths[i]
    Images[i,:,:,:] = np.asarray(Image.open(path).convert('RGB').resize((img_size,img_size)))
    Labels[i] = class_to_idx[path.split(os.sep)[-2]]
  return Images, Labels

# Load images as size 32x32; you can try with img_size = 64 to check if it improves the accuracy
img_size = 256
Images, Labels = LoadData(train_image_paths, img_size, class_to_idx)
# array of images and their labels, 0 for bad, 1 for good

#%%
def y_edge(img,y_threshold):
    mask = np.zeros(img.shape, dtype=bool)
    h,w = img.shape
    mask[np.abs(img) > y_threshold] = 1
    return mask
def x_edge(img,x_threshold):
    mask = np.zeros(img.shape, dtype=bool)
    h,w = img.shape
    mask[np.abs(img) > x_threshold] = 1
    return mask
def edge_detection(img,y_threshold,x_threshold):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_normalize = cv2.normalize(img_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    d = np.array([1,0,-1],ndmin=2);
    fx  = signal.convolve2d(img_gray_normalize,d,mode='same',boundary='symm');
    fy  = signal.convolve2d(img_gray_normalize,d.T,mode='same',boundary='symm');
    fy_mask = y_edge(fy,y_threshold)
    fx_mask = x_edge(fx,x_threshold)
    return fy_mask, fx_mask

fy_mask, fx_mask = edge_detection(Images[0],25,25)
#%%
def find_max_zero(fx_coloum,left,right):
    start = 0
    max_start = 0
    max = 0
    l = 0
    for i in range(left,right):
        if fx_coloum[i] < 5:
            if l==0:
                start=i
            l += 1
        else:
            if l>max:
                max = l
                max_start = start
            l = 0
    return max_start, max


def find_ROI_boundary(fy_mask, fx_mask, x_cutting, y_cutting):
    fy_row = np.array(np.sum(fy_mask, axis=1, keepdims=0))
    up = np.argmax(fy_row[0:150]) + y_cutting
    down = np.argmax(fy_row[150:]) + 150 - y_cutting

    fx_coloum = np.array(np.sum(fx_mask[up:down, :], axis=0))
    left_edge = np.argmax(fx_coloum[0:125])
    right_edge = np.argmax(fx_coloum[125:]) + 125
    start, length = find_max_zero(fx_coloum, left_edge, right_edge)
    left = start + x_cutting
    right = start + length - 1 - x_cutting

    return right, left, up, down
#%%
#for testing
for i in range(5):
    img = Images[i]
    fy_mask, fx_mask = edge_detection(img,25,25)
    right,left,up,down = find_ROI_boundary(fy_mask,fx_mask,10,20)
    plt.imshow(img, cmap="gray", alpha=0.5)
    plt.plot([0,255],[up,up],color='r')
    plt.plot([0,255],[down,down],color='r')
    plt.plot([left,left],[0,255],color='b')
    plt.plot([right,right],[0,255],color='b')
   # plt.savefig('processed_data//figure_{}.png'.format(i))
    plt.show()
#%%
def find_ROI(img):
    fy_mask, fx_mask = edge_detection(img,25,25)
    right,left,up,down = find_ROI_boundary(fy_mask,fx_mask,10,20)
    ROI = img[up:down,left:right]
    if ROI.size ==0: return ROI
    ROI_normalized = cv2.normalize(ROI, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ROI_normalized = cv2.resize(ROI_normalized, (64, 64), interpolation=cv2.INTER_LINEAR)
    return ROI_normalized



KNN_data = []
for i in range(len(Images)):
    img = Images[i]
    img_ROI = find_ROI(img)
    img_feature = np.array(np.mean(img_ROI,axis=1,keepdims=1))/255
    datapoint = np.hstack((Labels[i],img_feature.flatten()))
    KNN_data.append(datapoint)
KNN_data = np.array(KNN_data)
print(KNN_data.shape)
np.savetxt('KNN_database.csv',KNN_data,delimiter='\t',fmt='%.6f')