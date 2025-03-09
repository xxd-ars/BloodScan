from PIL import Image
img_path = './img/yolo_valid/13'
img = Image.open(img_path + '.jpg')
orig_w, orig_h = img.size
crop_left = 800
crop_top = 250
crop_width = 1216
crop_height = 1504 #1448

# crop_left = 250
# crop_top = 500
# crop_width = 900
# crop_height = 900

# cropped_img = img.crop((
#     crop_left,
#     crop_top,
#     crop_left + crop_width,
#     crop_top + crop_height
# ))
# cropped_img.save(img_path + '_cropped.jpg')


img = Image.open('./img/rawdata_demo_blue.jpg')
cropped_img = img.crop((
    crop_left,
    crop_top,
    crop_left + crop_width,
    crop_top + crop_height
))
cropped_img.save('./img/rawdata_demo_blue_.jpg')
