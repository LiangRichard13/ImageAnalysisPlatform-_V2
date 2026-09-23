import os
import cv2
import numpy as np
from datetime import datetime


# INPUT path
# image_folder = '/home/sstl/Disk1/data/silicon/datasets/3/origin_test'
# mask_folder = '/home/sstl/Disk1/data/silicon/datasets/3/gradcam_test'

image_folder = '/mnt/share/wry/silicon/1/images/'
mask_folder = '/mnt/share/wry/silicon/1/gradcams/'
# OUTPUT path
image_output_folder = '/mnt/share/wry/silicon/1/open_images'
mask_output = '/mnt/share/wry/silicon/1/normalized_cam.npz'  # 输出文件名

# 初始化一个字典来存储所有图像的数据和名称
data_dict = {}

# 遍历输入文件夹中的所有文件
filenames = os.listdir(image_folder)
filenames=sorted(filenames, key=lambda x: datetime.strptime(x, '%H:%M:%S.jpg'))[:3031]
# filenames = sorted(filenames, key=lambda x: int(x.split('_')[1].strip('Image').strip('.jpg')))


for filename in filenames:
    # if filename.lower().endswith(('.jpg')):
    # to gray value
    print(filename)

    # if filename.split('_')[1] in os.listdir(image_output_folder):
    if filename in os.listdir(image_output_folder):
        print(f'{filename} already exist')
        continue
    else:
        # gray value
        image = cv2.cvtColor(cv2.imread(os.path.join(image_folder, filename)), cv2.COLOR_BGR2GRAY)
        # resize
        image = cv2.resize(image, (800, 100))
        # histogram equalization
        # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4,4))
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
        image_histeq = clahe.apply(image)
        # save
        # cv2.imwrite(os.path.join(image_output_folder, filename.split('_')[1]),  image_histeq)
        cv2.imwrite(os.path.join(image_output_folder, filename),  image_histeq)

    if os.path.exists(os.path.join(mask_folder, filename)):
        # gray value
        mask = cv2.cvtColor(cv2.imread(os.path.join(mask_folder, filename)), cv2.COLOR_BGR2GRAY)
        # resize
        mask = cv2.resize(mask, (800, 100))
        # normalization
        mask_normalized = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    else:
        mask_normalized = np.ones_like(mask, dtype=np.float32)

    
    # data_dict[filename.split('_')[1]] = mask_normalized
    data_dict[filename] = mask_normalized

# 保存所有图像数据到一个npz文件中
np.savez_compressed(mask_output, **data_dict)
