import os
os.environ["TORCH_USE_FLASH_ATTENTION"] = "1"
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
# set TORCH_CUDNN_SDPA_ENABLED=1
# set TORCH_USE_FLASH_ATTENTION=1

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.backends.cuda.cudnn_sdp_enabled
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# np.random.seed(3)
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 182/255, 193/255, 0.6])
        # color = np.array([30/255, 144/255, 255/255, 0.6])
        
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    for datanumber in range(2, 6):
        ipath = "tests/segment/datasets/data_first/{}-B.png".format(datanumber)
        image = Image.open(ipath) # 1800*1200 'tests/segment/datasets/truck.jpg'
        image = np.array(image.convert("RGB"))

        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # plt.axis('on')
        # plt.show()

        sam2_checkpoint = "tests/segment/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)

        # input_point = np.array([[500, 375]])
        # input_label = np.array([1])
        input_point = np.array([[916, 128], [988, 1200], [985, 1388]])
        input_label = np.array([1, 1, 0])

        # input_point = np.array([[27,30,39],[220,180,50],[220,154,50],[180,180,150],[220,220,220]])
        # input_label = np.array([1, 1, 1, 1, 1])
        
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            # mask_input=mask_input[None, :, :], 
            # # 把此前的mask结果作为输入
            multimask_output=False)
        
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

        # mask_input = logits[np.argmax(scores), :, :]
        # masks, scores, _ = predictor.predict(
        #     point_coords=input_point,
        #     point_labels=input_label,
        #     mask_input=mask_input[None, :, :],
        #     multimask_output=False,)
        # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

        # input_box = np.array([425, 600, 700, 875])
        # masks, scores, _ = predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=input_box[None, :],
        #     multimask_output=False,)
        # show_masks(image, masks, scores, box_coords=input_box)