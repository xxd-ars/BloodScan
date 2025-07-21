# %% Parametres and import
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os, io, sys, cv2, json, re
from PIL import Image
import numpy as np

def calculate_iou(mask1, mask2):
    """
    计算两个二值掩码之间的 IoU。
    mask1 和 mask2 都是同样大小的 numpy 数组，
    前景像素值通常为 255（或非0），背景为 0。
    """
    # 如果前景值是 255，可以先转换为布尔值
    # 也可直接使用 mask1 > 0 的方式来获取前景区域
    m1 = (mask1 > 0)
    m2 = (mask2 > 0)

    intersection = np.logical_and(m1, m2).sum()  # 交集像素数
    union = np.logical_or(m1, m2).sum()          # 并集像素数

    if union == 0:
        # 如果两个掩码都完全没有前景，可能需要根据场景判断返回 1.0 还是 0.0
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou

def sort_points_by_angle(points):
    """
    根据点相对于质心的极角进行排序（逆时针）。
    适用于 points.shape == (N, 2) 且 N >= 3 的情况。
    """
    # 1. 计算质心
    center = np.mean(points, axis=0)  # [cx, cy]

    # 2. 计算相对于质心的向量和角度
    #   atan2(y, x) 返回(-π, π)，能区分象限，不会出现排序纠纷
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    # 3. 根据角度从小到大排序即可（默认升序）
    sorted_indices = np.argsort(angles)

    # 4. 返回排序后的点
    return points[sorted_indices]

project_dir= "/home/xin99/BloodScan/tests/yolo_seg"
project_dir= "./tests/yolo_seg"

epoch_num  = 10
model_name = f"train_blue_rawdata_1504_500_{epoch_num}epoch"
dataset_v  = "Blue-Rawdata-1504-500-1"

# if __name__ == '__main__':
# %% Loading model and test images
model = YOLO(project_dir + f"/runs/segment/{model_name}/weights/best.pt")
ipath = project_dir + f"/datasets/{dataset_v}/test/images/"

# %% ### Model evaluation with IOU and height difference
iou_list = []
height_upper_list = []
height_lower_list = []
height_upper_diff = []
height_lower_diff = []
height_upper_diff_percent = []
height_lower_diff_percent = []
detected_count = len(os.listdir(ipath))
for image_name in os.listdir(ipath):
    if not image_name.lower().endswith('.jpg'):
        continue
    
    ## testing use
    # image_name = '2022-03-28_143344_39_T5_2334_bmp_jpg.rf.03672cb0b717b22356ea1aa5976e7b24.jpg'
    try:
        json_nae = './data/rawdata/class1/' + image_name[:re.search(r'_bmp', image_name).start()] + '.json'
        with open(json_name, 'r') as f:
            data = json.load(f)
    except:
        json_name = image_name[:re.search(r'_bmp', image_name).start()] + '.json'
        json_name = json_name.replace("T5", "T3").rsplit("_", 1)[0] + f"_{int(json_name.rsplit('_', 1)[1].split('.')[0]) - 2}.bmp"
        json_name  = './data/rawdata/class2/' + json_name

        with open(json_name, 'r') as f:
            data = json.load(f)

    image_path = os.path.join(ipath, image_name)
    image = np.array(Image.open(image_path))
    annotated_image = image.copy()

    results = model(source = image, imgsz = 1504, 
                    device="cuda:0",
                    visualize = False, show = False, 
                    save = False, save_txt = False, save_conf = False, save_crop = False,
                    show_labels = False, show_conf = False, show_boxes = False, line_width = 2)
    cls_dict = results[0].names
    bloodzone2_id = [i for i, id in enumerate(list(results[0].boxes.cls.cpu().numpy())) if id == 1]
    if len(bloodzone2_id) > 0:
        points_list = []
        for i in bloodzone2_id:
            result = results[0][i]
            # mask = np.zeros(result.orig_shape, dtype=np.uint8)
            points = result.masks.xyn[0]
            points[:, 0] *= results[0].orig_shape[0]
            points[:, 1] *= results[0].orig_shape[1]
            points_list.append(points)

        points = np.vstack(points_list)
        points = points.astype(np.int32)
        height_upper = np.mean(np.sort(points[:, 1])[:2])
        height_lower = np.mean(np.sort(points[:, 1])[-2:])
        # points = sort_points_by_angle(points)
        # cv2.fillPoly(mask, [points], 255)
        mask = result.masks.data.cpu().numpy()[0]
        mask = cv2.resize(mask, (1504, 1504), interpolation=cv2.INTER_NEAREST)

        print('detected')
        cv2.polylines(annotated_image, points.reshape((-1, 1, 2)), isClosed=True, color=(0, 255, 0), thickness=5)
        # cv2.putText(annotated_image, str(class_id), (points[0][0], points[0][1] - 10), 
        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        ## true label
        true_mask = np.zeros(result.orig_shape, dtype=np.uint8)
        true_points = []
        for id, shape in enumerate(data["shapes"]):
            if id >= 2 and id <= 5:
                true_point = shape["points"][0]
                x, y = int(true_point[0]), int(true_point[1])
                # print(x, y)
                x, y = int((x - 800) * 1504/1216), int((y - 250) * 1)
                # x, y = int((x - 800) * 1500/1200), int((y - 250) * 1)
                # print(x, y)
                true_points.append([x, y])
                cv2.circle(annotated_image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)  # Red dot

        true_points = np.array(true_points, dtype=np.int32)
        true_points = sort_points_by_angle(true_points)
        true_height_upper = np.mean(np.sort(true_points[:, 1])[:1])
        true_height_lower = np.mean(np.sort(true_points[:, 1])[-1:])
        y_values = points[:, 1]

        cv2.fillPoly(true_mask, [true_points], 255)
        plt.imsave(f'./img/evaluation_rawdata_500/{image_name[:re.search(r'_bmp', image_name).start()]}.jpg', annotated_image)
        # plt.imshow(annotated_image)
        # plt.show()
        
        # print(f"Upper surface: \nTrue Value: {true_height_upper}, Predicted Value: {height_upper}, Diff: {true_height_upper - height_upper}, Diff %: {round((true_height_upper - height_upper)/true_height_upper*100, 2)}")

        # print(f"Lower surface: \nTrue Value: {true_height_lower}, Predicted Value: {height_lower}, Diff: {true_height_lower - height_lower}, Diff %: {round((true_height_lower - height_lower)/true_height_lower*100, 2)}")
        iou  = calculate_iou(true_mask, mask)
        # print(f"IOU: {iou}")

        height_upper_list.append(true_height_upper)
        height_lower_list.append(true_height_lower)
        height_upper_diff.append(abs(true_height_upper - height_upper))
        height_lower_diff.append(abs(true_height_lower - height_lower))
        height_upper_diff_percent.append(abs((true_height_upper - height_upper)/true_height_upper))
        height_lower_diff_percent.append(abs((true_height_lower - height_lower)/true_height_lower))

        iou_list.append(iou)
        # break
    else:
        print("No detection Area 1")
        detected_count = detected_count - 1
        plt.imsave(f'./img/evaluation_rawdata_500/{image_name[:re.search(r'_bmp', image_name).start()]}.jpg', annotated_image)

print('Bloodzone2 detection %', f"{round((detected_count/len(os.listdir(ipath)))*100, 2)}")
print('Average IOU', np.mean(iou_list))
print('Average upper surface difference', np.mean(height_upper_diff))
print('Average lower surface difference', np.mean(height_lower_diff))
print('Average upper surface difference %', f'{round(np.mean(height_upper_diff_percent)*100, 2)}%')
print('Average lower surface difference %', f'{round(np.mean(height_lower_diff_percent)*100, 2)}%')

# annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
# plt.imshow(annotated_img)
# plt.show()

fig, ax1 = plt.subplots(figsize=(9, 6))
bar_width = 0.4
x1, x2, x3, x4 = 0.2, 1., 1.8, 2.6
color='black'
ax1.bar(x=x1, height=detected_count/len(os.listdir(ipath)), capsize=5, color='C0', alpha=0.9, width=bar_width)
ax1.bar(x=x2, height=np.mean(iou_list), yerr=np.std(iou_list), capsize=5, color='C0', alpha=0.9, width=bar_width)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0, 1.])

ax2 = ax1.twinx()
ax2.bar(x=x3, height=np.mean(height_upper_diff), yerr=np.std(height_upper_diff), capsize=5, color='C0', alpha=0.9, width=bar_width)
ax2.bar(x=x4, height=np.mean(height_lower_diff), yerr=np.std(height_lower_diff), capsize=5, color='C0', alpha=0.9, width=bar_width)
ax2.set_ylabel('Diff (pixel over 1504 pixel)', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0, 10.])

plt.xticks([x1, x2, x3, x4], ['Recall', 'IoU', 'Upper level difference', 'Lower level difference'])
plt.title('Evaluation of yolo-seg on position accuracy', fontsize=10, pad=10)

ax1.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./img/evaluation_rawdata_500/barchart.png')
plt.show()