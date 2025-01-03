# -*- coding: iso-8859-1 -*-

import argparse
from ultralytics import YOLO
import os
import cv2

def main():
    parser = argparse.ArgumentParser(description="YOLO Segmentation Inference Script")
    parser.add_argument("--input", required=True, help="Path to the input image")
    parser.add_argument("--output", required=True, help="Path to save the annotated image")
    parser.add_argument("--weights", default="/home/xin99/BloodScan/tests/yolo_seg/weights/best_1125_1817.pt", help="Path to YOLO model weights")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on (e.g., 'cuda:0' or 'cpu')")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return

    model = YOLO(args.weights)

    print(f"Running inference on '{args.input}'...")
    results = model(
        source=args.input,
        imgsz=[768, 1024],
        device=args.device,
        visualize=False,
        show=False,
        save=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        line_width=2
    )
    results[0].save(args.output)
    # print(f"Saving annotated image to '{args.output}'...")
    # annotated_img = results[0].plot(
    #     labels=True,
    #     boxes=False,
    #     masks=True,
    #     probs=True,
    #     show=False,
    #     save=False,
    #     filename=None,
    #     color_mode='class'
    # )

    # annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    # os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # cv2.imwrite(args.output, annotated_img_bgr)

    print("Inference and saving completed successfully.")

if __name__ == "__main__":
    main()
