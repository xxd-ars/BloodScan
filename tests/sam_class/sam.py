from ultralytics import ASSETS, SAM, YOLO, FastSAM
import matplotlib.pyplot as plt
import cv2

datanumber = 2
test_file_path = "data/data_first/{}-B.png".format(datanumber)
image = cv2.imread(test_file_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load a model
model = SAM("tests/yolo/weights/sam2_b.pt")

# Profile SAM2-t, SAM2-b, SAM-b, MobileSAM
for file in ["sam_b.pt", "sam2_b.pt", "sam2_t.pt", "mobile_sam.pt"]:
    model = SAM(file)
    model.info()
    model(ASSETS)
    results = model(image)
    annotated_img = results[0].plot()
    plt.imsave("img/{}.png".format(file[:-3]), annotated_img)
    # plt.show()


# Display model information (optional)
# model.info()

# Run inference with bboxes prompt
# results = model("tests/yolo/datasets/bus.jpg", bboxes=[100, 100, 200, 200])
# results = model("data/data_first/2-B.png")
# results = model(image)

# # Run inference with single point
# results = model(points=[900, 370], labels=[1])

# # Run inference with multiple points
# results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

# # Run inference with multiple points prompt per object
# results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

# # Run inference with negative points prompt
# results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])

# annotated_img = results[0].plot()
# plt.imshow(annotated_img)
# plt.show()