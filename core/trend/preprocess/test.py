import os
import cv2
from datetime import datetime
import numpy as np
import pywt

# # image_folder = '/home/sstl/Disk1/data/silicon/datasets/1/images'
# image_folder = '/home/sstl/Disk1/data/silicon/datasets/3/origin_test'
# # 遍历输入文件夹中的所有文件
# filenames = os.listdir(image_folder)
# # filenames=sorted(filenames, key=lambda x: datetime.strptime(x, '%H:%M:%S.jpg'))[:3057]
# filenames = sorted(filenames, key=lambda x: int(x.split('_')[1].strip('Image').strip('.jpg')))

# # gray value
# image = cv2.cvtColor(cv2.imread(os.path.join(image_folder, filenames[0])), cv2.COLOR_BGR2GRAY)


# # histogram equalization
# clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(50, 50))
# image_histeq = clahe.apply(image)

# # filt = cv2.fastNlMeansDenoising(image_histeq, None, 27, 49, 49)

# filt = cv2.fastNlMeansDenoising(image_histeq, None, 3, 7, 21)


# # resize
# image = cv2.resize(filt, (800, 100))
# cv2.imwrite('filt-' + filenames[0], image)

img = cv2.imread('/home/sstl/Disk1/data/silicon/results/predgru/results-ontonet-2-mse/test_result/1/prediction_1.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(img, (4096, 500))
# cv2.imwrite('prediction_1.jpg', image)


# 计算直方图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 将直方图数组转换为可绘制的图像
hist_img = np.zeros((300, 256, 1), dtype=np.uint8)
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
for x, h in enumerate(hist):
    cv2.line(hist_img, (x, 300), (x, 300 - int(h)), 255, 1)

# 保存直方图图像
cv2.imwrite('histogram.png', hist_img)


# gamma = 0.9
# invGamma=1.0// gamma
# table=np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
# image_gamma_corrected = cv2.LUT(filt, table)


# equ = cv2.equalizeHist(filt)





