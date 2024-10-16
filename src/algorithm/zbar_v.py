# -*- coding:utf-8 -*-
'''
这个脚本的主要功能是批量处理文件夹中的图片，对每张图片进行二维码解码，并将结果存储在Excel文件中。代码还包括生成统计信息的功能，但这些部分（如绘图）已被注释。

主要功能
获取二维码结果

    函数: get_qrcode_result(image_input, binary_max=255, binary_step=5)
    功能: 从输入图像中提取二维码信息，返回解码的二维码数据。
    参数:
    image_input: 输入的图像数据。
    binary_max: 二值化的最大值，默认为255。
    binary_step: 每次递增的二值化步长，默认为5。
    返回: 解码后的二维码数据字符串。

解码文件夹中的所有图像

    函数: decode_images_in_folder(folder_path)
    功能: 扫描指定文件夹中的所有图片，解码二维码，并生成一个Excel文件保存结果。
    参数:
    folder_path: 要扫描的文件夹路径。
    返回: 无'''
import datetime
import time
from pathlib import Path
import numpy as np
import cv2
from pyzbar import pyzbar
import pandas as pd
import re
import argparse
import imutils
# import matplotlib.pyplot as plt
from tqdm import tqdm  


def get_qrcode_result(image_input, binary_max=255, binary_step=5):
    """
    获取二维码的结果、二值化值和解码时间
    :param image_input: 输入图片数据
    :param binary_max: 二值化的最大值
    :param binary_step: 每次递增的二值化步长
    :return: pyzbar 预测的结果, 二值化值, 解码时间
    """
    # image1 = image_input
    # start_time = time.time()
    number = 1
    
    # 把输入图像灰度化
    if len(image_input.shape) >= 3:
        image_input = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)

    # 获取自适配阈值
    binary, _ = cv2.threshold(image_input, 0, 255, cv2.THRESH_OTSU)

    # 二值化递增检测
    while binary < binary_max:
        _, mat = cv2.threshold(image_input, binary, 255, cv2.THRESH_BINARY)
        # _, mat = cv2.threshold(image1, binary, 255, cv2.THRESH_BINARY)
        res = pyzbar.decode(mat)
        # if len(res) == 1:
        if len(res) >0:
            barcode_data = res[0].data.decode("utf-8")
            if barcode_data.isdigit() and len(barcode_data) == 12:
                # return res[0].data.decode("utf-8"), binary, time.time() - start_time,number
                return res[0].data.decode("utf-8")
            else :
                if barcode_data.isdigit() and len(barcode_data) == 11:
                    # return res[0].data.decode("utf-8"), binary, time.time() - start_time,number
                    return res[0].data.decode("utf-8")

        binary += binary_step
        number += 1

    # return "", binary, time.time() - start_time,number
    return ""



def decode_images_in_folder(folder_path):
    """
    解码文件夹中所有符合条件的图片,并生成Excel文件
    :param folder_path: 文件夹路径
    """
    results = []
    binary_steps = range(1, 21)  # 二值化步长范围为1到20
    number_avgs = []
    time_avgs = []
    number_maxs = []

    for binary_step in tqdm(binary_steps, desc="Processing"):  # 添加进度条
        number_sum = 0
        time_sum = 0
        max_number = 0
        for image_file in tqdm(Path(folder_path).glob("*-A.png"), desc="Images"):  # 添加进度条
            # 获取图片编号
            match = re.search(r'(\d+)-A.png', image_file.name)
            if match:
                image_number = match.group(1)
            else:
                image_number = "Unknown"

            image = cv2.imread(str(image_file))
            if image is not None:
                barcode_data, binary_value, decode_time,number = get_qrcode_result(image,binary_step = binary_step)
                if barcode_data:
                    results.append({"Image Number": image_number, "Barcode": barcode_data, 
                                    "Binary Value": binary_value, "Decode Time": decode_time,
                                    "number":number})

                else:
                    results.append({"Image Number": image_number, "Barcode": "nope", 
                                    "Binary Value": binary_value, "Decode Time": decode_time,
                                    "number":number})
                
                number_sum += number
                time_sum += decode_time
                max_number = max(max_number, number)

        if len(results) > 0:
            number_avg = number_sum / len(results)
            time_avg = time_sum / len(results)
        else:
            number_avg = 0
            time_avg = 0
        
        number_avgs.append(number_avg)
        time_avgs.append(time_avg)
        number_maxs.append(max_number)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(binary_steps, number_avgs, marker='o')
    # plt.xlabel('Binary Step')
    # plt.ylabel('Average Number')
    # plt.title('Average Number vs Binary Step')
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(binary_steps, time_avgs, marker='o')
    # plt.xlabel('Binary Step')
    # plt.ylabel('Average Decode Time')
    # plt.title('Average Decode Time vs Binary Step')
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()
    
    # # 将结果转换为DataFrame
    # df = pd.DataFrame(results)
    # # 保存为Excel文件
    # df.to_excel("decoded_images7.xlsx", index=False)

# if __name__ == "__main__":
#     folder_path = "data_first"
#     decode_images_in_folder(folder_path)
