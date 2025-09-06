from PIL import Image
import argparse
import os

def rotate_image(input_path, output_path, angle):
    # 打开图片
    with Image.open(input_path) as img:
        # 顺时针旋转：使用负角度
        rotated = img.rotate(-angle, expand=True)
        # 保存输出
        rotated.save(output_path)
        print(f"图像已保存到：{output_path}")

def main():

    input_path = "dual_yolo/evaluation_results_aug_0_1_2_70conf/id/2022-03-28_103204_17_T5_2412_6_evaluation.jpg"
    angle = -10
    output_path = "output_rotate.png"

    rotate_image(input_path, output_path, angle)

def generate_output_path(input_path, angle):
    base, ext = os.path.splitext(input_path)
    return f"{base}_rotated_{int(angle)}{ext}"

if __name__ == "__main__":
    main()