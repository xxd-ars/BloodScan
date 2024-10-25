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
#loading and constructing related functions
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

def find_ROI(img,img_blue=None):
    fy_mask, fx_mask = edge_detection(img,25,25)
    right,left,up,down = find_ROI_boundary(fy_mask,fx_mask,10,20)
    ROI = img[up:down,left:right]
    if ROI.size ==0: return ROI
    ROI_normalized = cv2.normalize(ROI, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ROI_normalized = cv2.resize(ROI_normalized, (64, 64), interpolation=cv2.INTER_LINEAR)

    ROI_blue = None
    if img_blue.any()!=None:
        ROI_blue = img_blue[up:down,left:right]
        ROI_blue = cv2.resize(ROI_blue, (64, 64), interpolation=cv2.INTER_LINEAR)
    return ROI_normalized,ROI_blue

#%%
def Load_KNN_datasets(data_file):
    data = np.loadtxt(data_file,delimiter='\t')
    labels = data[:,0]
    features = data[:,1:]
    return labels,features


def KNN_predict(KNN_features,KNN_Labels,X,k,img,img_blue = None):
    dists = np.linalg.norm(KNN_features-X,axis=1,keepdims=0)
    top_k_index = np.argsort(dists)[:k]
    closest_y = KNN_Labels[top_k_index]
    vote = Counter(closest_y)
    count = vote.most_common()
    if count[0][1] != k/2: return count[0][0], top_k_index
    else:
        return KNN_predict(KNN_features,KNN_Labels,X,k+1)
    """
    #introducing BI info
    if count[0][1] > 0.7*k or img_blue = None: return count[0][0], top_k_index
    _,b = find_layer(img,img_blue)
    if b>0.75: return 1, None
    else: return 0, None
    """

#%%
def find_layer(img,img_blue):
    fy_mask, fx_mask = edge_detection(img,25,25)
    right,left,up,down = find_ROI_boundary(fy_mask,fx_mask,10,20)
    ROI = img_blue[up:down,left:right]
    ROI_normalized = cv2.normalize(ROI, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ROI_gray = cv2.cvtColor(ROI_normalized, cv2.COLOR_BGR2GRAY)
    ROI_flatten = np.array(np.mean(ROI_gray,axis=1,keepdims=1))
    ROI_tile = np.tile(ROI_flatten, (1, 180))
    # print(ROI_tile.shape)
    # plt.imshow(ROI_tile,cmap = 'gray')
    # plt.show()
    # ROI_diff = ROI_flatten[1:] - ROI_flatten[:-1]
    # print(np.argmax(ROI_diff>20))
    ROI_peak = 2*ROI_flatten[1:-1] - ROI_flatten[:-2] - ROI_flatten[2:]
    h = np.argmax(ROI_peak)+up
    img_layer = img.copy()
    b = np.mean(img_layer[h,:])
    img_layer[h-1:h+1,:] = [0,255,0]
    # plt.imshow(img_layer)
    # plt.show()
    return  img_layer, b


#%%
def predict(test_RI,test_BI=None,KNN_data = "KNN_database.csv",k=11):
    #preparing
    KNN_labels, KNN_features = Load_KNN_datasets(KNN_data)
    ROI, ROI_blue = find_ROI(test_RI, test_BI)
    X = np.array(np.mean(ROI,axis=1,keepdims=1))/255
    X = X.reshape(-1)
    #predict
    pred_label, nearest_index = KNN_predict(KNN_features, KNN_labels, X, k, test_RI, test_BI)
    #output
    comment = "good" if pred_label == 1 else "bad due to ..."
    img_showlayer = None
    if pred_label ==1 and test_BI.any() != None:
        img_showlayer,b = find_layer(test_RI,test_BI)
    return comment, img_showlayer


#%%
Regular_Image_Path = "data_V1_test\\good\\figure_831_T3.png"
Bluelight_Image_path = "data_V1_test\\good\\figure_831_T5.png"
img_size = 256

test_RI = np.asarray(Image.open(Regular_Image_Path).convert('RGB').resize((img_size,img_size)))
test_BI = np.asarray(Image.open(Bluelight_Image_path).convert('RGB').resize((img_size,img_size)))

    #%%

comment,img_layer =  predict(test_RI,test_BI)
print(comment)
plt.imshow(img_layer)
plt.show()

# X = np.array(np.mean(ROI, axis=1, keepdims=1)) / 255
# X = X.reshape(-1)
# pred_label, nearest_index = predict(KNN_datasets, Train_Labels, X, k)
# total += 1
# if pred_label == Test_Labels[i]: correct += 1