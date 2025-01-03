from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import cv2

# 创建 Flask 应用
app = Flask(__name__)

# 加载模型（服务启动时只加载一次）
model = YOLO("/home/xin99/BloodScan/tests/yolo_seg/weights/best_1125_1817.pt")
print("Model loaded successfully.")

# 推理端点
@app.route('/infer', methods=['POST'])
def infer():
    try:
        # 获取 JSON 数据
        data = request.json
        input_image = data['input_image']
        output_image = data['output_image']

        # 检查输入文件是否存在
        if not os.path.exists(input_image):
            return jsonify({"status": "error", "message": f"Input file '{input_image}' not found."}), 400

        # 运行推理
        print(f"Running inference on {input_image}...")
        results = model(
        conf = 0.6,
        source=input_image,
        imgsz=[768, 1024],
        device="cuda:0",
        visualize=False,
        show=False,
        save=True,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=False,
        show_conf=False,
        show_boxes=False,
        line_width=2)

        results[0].save(output_image)

        
        return jsonify({"status": "success", "output_image": output_image})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # 服务运行在 0.0.0.0:5000
    app.run(host="0.0.0.0", port=5000)
