import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from dual_modal_yolo import DualModalYOLO


def parse_args():
    parser = argparse.ArgumentParser(description="使用双模态YOLO11分割模型进行推理")
    parser.add_argument('--white-img', type=str, required=True, help='白光图像路径')
    parser.add_argument('--blue-img', type=str, required=True, help='蓝光图像路径')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--img-size', type=int, default=1024, help='推理图像大小')
    parser.add_argument('--device', type=str, default='cuda:0', help='推理设备')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--fusion-type', type=str, default='transformer', 
                        choices=['add', 'concat', 'transformer'], help='特征融合类型')
    parser.add_argument('--save-dir', type=str, default='runs/predict', help='结果保存路径')
    parser.add_argument('--view', action='store_true', help='是否显示推理结果')
    return parser.parse_args()


def preprocess_image(img_path, img_size=1024):
    """
    预处理图像
    Args:
        img_path: 图像路径
        img_size: 图像大小
    Returns:
        img: 预处理后的图像张量，形状为(1, 3, H, W)
        orig_img: 原始图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    orig_img = img.copy()
    
    # 转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 缩放图像
    img = cv2.resize(img, (img_size, img_size))
    
    # 转换为张量
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)  # 添加批次维度
    
    return img, orig_img


def draw_results(image, bboxes, masks, labels, scores, class_names, colors=None):
    """
    在图像上绘制检测和分割结果
    Args:
        image: 原始图像
        bboxes: 边界框坐标 [x1, y1, x2, y2]
        masks: 分割掩码
        labels: 类别标签
        scores: 置信度分数
        class_names: 类别名称列表
        colors: 颜色列表，若为None则随机生成
    Returns:
        result_img: 绘制结果后的图像
    """
    # 复制原始图像
    result_img = image.copy()
    h, w = image.shape[:2]
    
    # 如果没有提供颜色，则随机生成
    if colors is None:
        np.random.seed(42)
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(class_names))]
    
    # 遍历每个检测结果
    for i, (bbox, mask, label, score) in enumerate(zip(bboxes, masks, labels, scores)):
        # 获取类别颜色
        color = colors[label]
        
        # 绘制边界框
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制类别和置信度
        text = f"{class_names[label]}: {score:.2f}"
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(result_img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(result_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # 绘制分割掩码
        if mask is not None:
            # 将掩码转换为和原图相同大小
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            # 创建彩色掩码
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
            colored_mask[mask > 0.5] = color
            
            # 将彩色掩码叠加到结果图像上
            alpha = 0.5  # 透明度
            cv2.addWeighted(colored_mask, alpha, result_img, 1 - alpha, 0, result_img)
    
    return result_img


def main(args):
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print(f"加载模型权重: {args.weights}")
    model = DualModalYOLO(
        white_model_path='yolo11x-seg.pt',
        blue_model_path='yolo11x-seg.pt',
        fusion_type=args.fusion_type
    )
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 预处理白光和蓝光图像
    print(f"预处理图像...")
    white_img, orig_white_img = preprocess_image(args.white_img, args.img_size)
    blue_img, orig_blue_img = preprocess_image(args.blue_img, args.img_size)
    
    # 将图像移动到指定设备
    white_img = white_img.to(device)
    blue_img = blue_img.to(device)
    
    # 推理
    print(f"开始推理...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(white_img, blue_img)
    
    # 计算推理时间
    inference_time = time.time() - start_time
    print(f"推理完成，耗时: {inference_time:.4f}秒")
    
    # 后处理结果
    results = post_process(outputs, orig_white_img.shape[:2], 
                          conf_thres=args.conf_thres, 
                          iou_thres=args.iou_thres)
    
    # 绘制结果
    class_names = ['background', 'bloodzone', 'other']  # 根据实际类别调整
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # 蓝色、绿色、红色
    
    result_img = draw_results(
        orig_white_img, 
        results['boxes'], 
        results['masks'], 
        results['labels'], 
        results['scores'], 
        class_names, 
        colors
    )
    
    # 保存结果
    result_path = save_dir / f"result_{Path(args.white_img).stem}.jpg"
    cv2.imwrite(str(result_path), result_img)
    print(f"结果已保存到: {result_path}")
    
    # 显示结果
    if args.view:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title("检测结果")
        plt.axis('off')
        plt.show()
    
    # 返回结果字典
    return {
        'boxes': results['boxes'],
        'masks': results['masks'],
        'labels': results['labels'],
        'scores': results['scores'],
        'inference_time': inference_time
    }


def post_process(outputs, orig_size, conf_thres=0.25, iou_thres=0.45):
    """
    后处理模型输出结果
    Args:
        outputs: 模型输出
        orig_size: 原始图像大小 (h, w)
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
    Returns:
        results: 处理后的结果，包含boxes, masks, labels, scores
    """
    # 这里需要根据模型的输出格式进行相应的后处理
    # 以下是一个示例，实际实现需要根据YOLO11模型的输出格式进行调整
    
    # 假设outputs包含检测结果和分割结果
    # det_output: [batch, num_dets, 6] (x1, y1, x2, y2, conf, cls)
    # seg_output: [batch, num_dets, h, w] (分割掩码)
    
    # 提取边界框、置信度和类别
    preds = outputs[0]  # 取第一个批次的结果
    
    # 应用NMS
    keep_idxs = nms(preds[:, :4], preds[:, 4], iou_thres)
    preds = preds[keep_idxs]
    
    # 过滤低置信度预测
    conf_mask = preds[:, 4] >= conf_thres
    preds = preds[conf_mask]
    
    # 如果没有检测结果，返回空结果
    if len(preds) == 0:
        return {
            'boxes': [],
            'masks': [],
            'labels': [],
            'scores': []
        }
    
    # 提取边界框坐标
    boxes = preds[:, :4].cpu().numpy()
    
    # 调整边界框大小到原始图像
    h, w = orig_size
    img_h, img_w = args.img_size, args.img_size
    scale_x, scale_y = w / img_w, h / img_h
    
    # 缩放边界框到原始图像大小
    boxes[:, 0] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 2] *= scale_x
    boxes[:, 3] *= scale_y
    
    # 提取置信度分数
    scores = preds[:, 4].cpu().numpy()
    
    # 提取类别标签
    labels = preds[:, 5].cpu().numpy().astype(int)
    
    # 提取分割掩码
    masks = []
    for i, box in enumerate(boxes):
        # 如果模型输出包含分割掩码，则处理
        if len(outputs) > 1 and i < len(outputs[1]):
            mask = outputs[1][i].cpu().numpy()
            # 调整掩码大小到原始图像
            mask = cv2.resize(mask, (w, h))
            masks.append(mask)
        else:
            # 如果没有分割掩码，使用边界框创建一个矩形掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)
    
    return {
        'boxes': boxes,
        'masks': masks,
        'labels': labels,
        'scores': scores
    }


def nms(boxes, scores, iou_threshold=0.45):
    """
    非极大值抑制
    Args:
        boxes: 边界框坐标 [N, 4] (x1, y1, x2, y2)
        scores: 置信度分数 [N]
        iou_threshold: IOU阈值
    Returns:
        keep: 保留的索引
    """
    # 获取排序索引
    order = torch.argsort(scores, descending=True)
    
    keep = []
    while order.numel() > 0:
        # 取出得分最高的框
        i = order[0].item()
        keep.append(i)
        
        # 如果只剩下一个框，则结束
        if order.numel() == 1:
            break
        
        # 计算最高得分框与其他框的IoU
        xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
        yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
        xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
        yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        
        iou = inter / (area1 + area2 - inter)
        
        # 保留IoU小于阈值的框
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return torch.tensor(keep, dtype=torch.long)


if __name__ == "__main__":
    args = parse_args()
    results = main(args) 