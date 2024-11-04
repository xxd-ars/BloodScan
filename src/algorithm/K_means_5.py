'''该模块主要用于从采血管图像中识别不同的血液分层，并进行相应的分类和计算。主要步骤包括图像预处理、聚类分析、区域检测、类型检查和深度计算。

主要功能
图像预处理

    函数: paral_read(img, ru, rd, w)
    功能: 将输入图像进行平行读取和裁剪，返回裁剪后的图像。
    参数:
    img: 输入图像。
    ru: 上部裁剪坐标。
    rd: 下部裁剪坐标。
    w: 裁剪宽度。
    返回: 裁剪后的图像。
KMeans 聚类

    函数: KM(img, n_cluster, inital_center)
    功能: 对图像进行 KMeans 聚类，返回分割后的图像。
    参数:
    img: 输入图像。
    n_cluster: 聚类数量。
    inital_center: 初始中心点。
    返回: 分割后的图像。
区域检测

    函数: region_check(segmented_image, index)
    功能: 检测分割图像中的特定区域，返回该区域。
    参数:
    segmented_image: 分割后的图像。
    index: 区域索引。
    返回: 特定区域。
类型检查

    函数: type_check(segmented_image, string)
    功能: 检查图像的类型（如血浆层、红细胞层等），并返回检查结果。
    参数:
    segmented_image: 分割后的图像。
    string: 描述字符串。
    返回: 检查结果。
层次检测和深度计算

    函数: find_layer(region, threshold) 和 plasma_check(region, threshold, h_layer)
    功能: 检测血液层次，并计算白膜层的深度。
加载数据

    函数: load_data(index)
    功能: 加载指定索引的图像数据，并进行预处理。
测试函数

    函数: test(k)
    功能: 对指定索引的图像进行测试，输出分层识别结果和白膜层深度。'''
#%%
from sklearn.cluster import KMeans
import math
import shutil
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
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
# %%
def paral_read(img,ru,rd,w):
    # pa_readral(img, (300, 840), (1290, 900), 160) (y, x)
    h1, w1 = ru
    h2, w2 = rd
    img_paral = np.zeros((h2 - h1, w, 3), dtype=np.uint8)
    for h in range(h1, h2):
        start = int((w2-w1)/(h2-h1)*(h-h1)+w1)
        img_paral[h-h1,:,:] = img[h,start:start+w,:]
    return img_paral
#%%
def KM(img,n_cluster,inital_center):
    """
    img: 输入的图像数据，通常是一个三维数组（高度、宽度、颜色通道）。
    n_cluster: 要划分的簇的数量，即目标分割的类别数。
    inital_center: K-Means 算法的初始聚类中心，作为算法的起始点。
    """
    pixels = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_cluster,init=inital_center, n_init=1, random_state=0,max_iter=100)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    print("labels: ", set(labels))
    segmented_image = labels.reshape(img.shape[0], img.shape[1])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(segmented_image, cmap='viridis')
    ax[1].set_title('Segmented Image')
    ax[1].axis('off')

    plt.show()
    return segmented_image

#%%
def region_check(segmented_image,index):

    data = (segmented_image == index)
    n_point = np.sum(data) # 目标区域(index)像素点的数量
    
    labeled_array, num_features = measure.label(data, connectivity=2, return_num=True)
    # 对二值图像进行连通区域标记
    # num_features连通区域的数量, labeled_array每个像素被赋予一个连通区域的标签从 1 开始

    # 查找主要的连通区域
    major_feature = 0
    for feature in range(1,num_features+1):
        n_current_point = np.sum(labeled_array==feature)
        if n_current_point > 0.7*n_point and n_current_point>2000:
            major_feature = feature
            break

    if major_feature == 0:
        return None
    else:
        return labeled_array==major_feature

def avg_hw(binary_img):
    h_avg = np.mean(np.argwhere(binary_img)[:,0]).astype(int)
    w_avg = np.mean(np.argwhere(binary_img)[:,1]).astype(int)
    return h_avg,w_avg


def gel_check(segmented_image,h1,h2):
    data = (segmented_image == 3)
    # if major area exist or not
    labeled_array, num_features = measure.label(data, connectivity=2, return_num=True)
    major_feature = 0
    for feature in range(1,num_features+1):
        region = labeled_array==feature
        n_current_point = np.sum(region)
        if n_current_point>2000:
            h3 = avg_hw(region)[0]
            if h1>h3 and h2<h3:
                return region
    return None


def type_check(segmented_image,string):
    comment = ""
    h1,h2,h3= None,None,None
    region1 = region_check(segmented_image,0)
    if region1 is not None:
        h1,_ = avg_hw(region1)
    else:
        string = '红细胞层不存在'
        # print("红细胞层不存在")
    region2 = region_check(segmented_image,1)
    region3 = region_check(segmented_image,2)
    if region2 is not None and region3 is not None:
        h2,_ = avg_hw(region2^region3)
        region2 = region2^region3
    else:
        if region2 is None and region3 is None:
            string = '血浆层不存在'
        # print("血浆层不存在")
        else:
            region2 = region3*1 if region2 is None else region2
            h2,_ = avg_hw(region2)
    region4 = gel_check(segmented_image,h1,h2)
    if region4 is not None:
        string = '凝胶样本'
        # print("坏样本，凝胶存在，不判断")
        return False,(region1,region2,region3),string
    return True,(region1,region2),string

def find_layer(region,threshold):
    for h_row in range(region.shape[0]):
        row = region[h_row,:]
        if np.sum(row)>threshold:
            h_layer = h_row
            break
    return h_layer

def plasma_check(region,threshold,h_layer):
    for h_row in range(region.shape[0]):
        row = region[h_row,:]
        if np.sum(row)>threshold and h_row>h_layer:
            # print(h_row,np.sum(row))
            return False
    return True

#%%
def load_data(index):
    if index<=200:
        test_file_path = "../../data/data_first/{}-B.png".format(index)
        w_file, h_file = (2048, 1536) # 定义图像大小
        img = np.asarray(Image.open(test_file_path).convert('RGB').resize((w_file, h_file)))
        img_paral = paral_read(img, (300, 840), (1290, 900), 160)
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(img_paral)
        # plt.show()
        return img,img_paral,300
    else:
        test_file_path = "../../data/data_first/{}-B.png".format(index)
        w_file, h_file = (2048, 1536)
        pos_base = np.array([400, 1130, 910, 1060])
        img = np.asarray(Image.open(test_file_path).convert('RGB').resize((w_file, h_file)))
        img_clip = img[pos_base[0]:pos_base[1], pos_base[2]:pos_base[3]]
        return img,img_clip,400

def test(k):
    start_time = time.time()
    n_clusters = 5
    initial_center = np.array([[27,30,39],[220,180,50],[220,154,50],[180,180,150],[220,220,220]])

    #choosing test pic number

    string  = None
    h_depth = 0
    img_draw = False
    img,ROI,h_intrinsic = load_data(k)
    segmented_image = KM(ROI,n_clusters,'k-means++')
    draw,regions,string = type_check(segmented_image,string)
    seg2_image = np.ones(segmented_image.shape)*3
    seg2_image[regions[0]] = 0
    seg2_image[regions[1]] = 1
    if len(regions)==3 : seg2_image[regions[2]] = 2
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].imshow(img)
    # ax[0].set_title('Original Image')
    # ax[0].axis('off')
    # ax[1].imshow(seg2_image, cmap='viridis')
    # ax[1].set_title('Segmented Image')
    # ax[1].axis('off')
    # plt.show()
    if draw:
        h_layer = find_layer(regions[0],0.2*160)
        img_draw = ROI*1
        img_draw[h_layer-2:h_layer+2,:] = [0,255,0]
        # plt.imshow(img_draw)
        # plt.show()
        if plasma_check(regions[1],20,h_layer):
            string = '血浆样本'
            # print("好样本，血浆样本")
            # img = Image.fromarray(img)
            # img.save("..\\dataset\\dataset2\\data_1\\{}.png".format(k))
        else:
            string = '血清样本'
            # print("坏样本，血清样本")
            # img = Image.fromarray(img)
            # img.save("..\\dataset\\dataset2\\data_2\\{}.png".format(k))
        if k<=200:
            h_depth = (h_layer+230)*(7.5/160)
        else:
            h_depth = (h_layer+50)*(10.0/150)
        print("白膜层的深度为{:.2f}mm".format(h_depth))
    # else:
    #     img = Image.fromarray(img)
    #     img.save("..\\dataset\\dataset2\\data_3\\{}.png".format(k))
            
    print(time.time()-start_time)
    return draw,segmented_image,img_draw,string,h_depth
# %%
