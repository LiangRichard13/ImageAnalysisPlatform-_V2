import cv2 as cv
import numpy as np
import json
import math
import os
import base64
from datetime import datetime

class Cluster:
    def __init__(self, centroid):
        # 中心为第一个加入的point，准备改成质心
        self.centroid = centroid
        #((x,y), w, h, degree)
        self.members = []
    
    def distance(self, point, l2, l3):
        d1 = np.linalg.norm(point[0] - self.centroid[0])
        d2 = ( self.centroid[1] * self.centroid[2] + point[1] * point[2] )/2
        d3 = abs(point[3] - self.centroid[3])
        d = (d1 + l3*d3)/(l2 * d2)
        return d


class SinglePassClustering:
    def __init__(self, threshold, l2, l3):
        self.threshold = threshold
        self.clusters = []
        self.l2 = l2 # hyperparameter of avg area
        self.l3 = l3 # hyperparameter of angle difference
    
    def find_nearest_cluster(self, point):
        nearest_cluster = None
        min_distance = float('inf')
        
        for cluster in self.clusters:
            distance = cluster.distance(point, self.l2, self.l3)
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = cluster
        
        return nearest_cluster
    
    def add_point(self, point):
        nearest_cluster = self.find_nearest_cluster(point)
        
        if nearest_cluster is not None and nearest_cluster.distance(point, self.l2, self.l3) <= self.threshold:
            nearest_cluster.members.append(point)
        else:
            new_cluster = Cluster(point)
            self.clusters.append(new_cluster)
    
    def get_clusters(self):
        return self.clusters

def process_image(image, low_threshold=50, high_threshold=150, dilation_size=5, block_size=51, c=-9):
    # 读取并转换为灰度图
    # image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    """image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)"""

    # 使用Canny边缘检测器
    edges = cv.Canny(image, low_threshold, high_threshold)

    # 对边缘进行形态学膨胀操作以合并冗余的线条
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (dilation_size, dilation_size))
    dilated_edges = cv.dilate(edges, kernel)

    # 寻找轮廓（此时边缘已经是闭合的区域）
    contours, _ = cv.findContours(dilated_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 创建一个结果图像
    result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # 遍历轮廓，对每个区域进行处理
    for contour in contours:
        # 将轮廓内部填充为白色
        cv.drawContours(result, [contour], 0, (255, 255, 255), -1)

    # 将结果图像转换为灰度图像
    gray_result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    # 获取黑色区域的坐标
    black_pixels = np.where(gray_result == 0)

    # 使用自适应阈值处理计算局部阈值
    adaptive_threshold = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, c)

    # 遍历每个黑色像素
    for y, x in zip(*black_pixels):
        # 检查该黑色像素对应自适应阈值图像中的值是否为白色（255）
        if adaptive_threshold[y, x] == 255:
            # 如果是白色，将结果图像中对应位置设置为白色
            result[y, x] = (255, 255, 255)
    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    # TODO
    return result

def calculate_voltage_from_slope(avg_deg):
    """
    根据斜率角度计算电压值
    Args:
        avg_deg: 平均角度（0-180度）
    Returns:
        float: 电压值，范围-5到+5伏特
    """
    # 将角度转换为斜率（正切值）
    slope = math.tan(math.radians(avg_deg))
    
    # 线性映射：电压 = 0.05 × 斜率，斜率范围±100对应电压范围±5
    voltage = 0.05 * slope
    
    # 限制电压在±5伏特范围内
    voltage = max(min(voltage, 5.0), -5.0)
    
    return round(voltage, 2)

# 这里输出斜率数值
def hough(image, threshold=50, minLineLength=50, maxLineGap=20): # 使用你极端的参数
    
    # ------------------- 调试步骤 1: 检查输入图像 -------------------
    if image is None:
        print("错误：传入的'image'对象是 None。")
        return None, 0.0
    
    print(f"--- 开始检测 ---")
    print(f"输入图像的形状: {image.shape}, 数据类型: {image.dtype}")
    # 检查图像是否大部分是黑色的
    if np.max(image) < 50: # 假设像素值范围是0-255
        print("警告：输入图像非常暗，可能没有足够的对比度。")
    # ----------------------------------------------------------------

    image_copy = np.copy(image)
    
    if len(image_copy.shape) == 2 or image_copy.shape[2] == 1:
        gray = image_copy
        image_color = cv.cvtColor(image_copy, cv.COLOR_GRAY2BGR)
    else:
        gray = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
        image_color = image_copy

    # ------------------- 调试步骤 2: 检查Canny的输入和输出 -------------------
    print(f"灰度图的形状: {gray.shape}, 数据类型: {gray.dtype}")
    
    # 使用你原来的Canny参数
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    
    # 关键检查点：计算Canny检测到了多少个边缘点
    edge_pixel_count = cv.countNonZero(edges)
    print(f"Canny检测到的边缘点数量: {edge_pixel_count}")

    if edge_pixel_count == 0:
        print("诊断：Canny没有检测到任何边缘点。这是找不到线条的根本原因。")
        return image_color, 0.0

    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
    
    if lines is None:
        print(f"诊断：HoughLinesP依然没有检测到线条。")
        print(f"使用的参数: threshold={threshold}, minLineLength={minLineLength}, maxLineGap={maxLineGap}")
        return image_color, 0.0
    
    cnt_line = 0
    sum_deg = 0
    for line in lines:
        cnt_line += 1
        x1,y1,x2,y2 = line[0]
        if x1 == x2:
            deg = 90.0
        else:
            deg = math.degrees(np.arctan((y2-y1) / (x2-x1)))
            if deg < 0:
                deg += 180
        sum_deg += deg
        cv.line(image_color,(x1,y1),(x2,y2),(0,255,0),2)
    avg_deg = sum_deg/cnt_line if cnt_line > 0 else 0.0
    
    print(f"成功检测到 {cnt_line} 条线。")
    return image_color, avg_deg

def drawContours(image):
    # return image, ( center(x,y), (w,h), angle, box(最小矩形框，斜的), [x1,y1], [x2,y2](矩形左上角与右下角坐标))
    image = np.copy(image)
    """img1= cv.cvtColor(image,cv.COLOR_BGR2GRAY)"""

    # img1 = cv.Canny(gray, 100, 200)
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = 0

    wrinkle_list = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # cv.line(hough2, (x, y), (x, y + h), (255, 0, 0), 2, 5)    # 左侧
        # cv.line(hough2, (x + w, y), (x + w, y + h), (255, 0, 0), 2, 5)    # 右侧线
        if w>10:#宽度大于5才画
            cv.rectangle(image, (x,y),(x+w,y+h),(0,255,0),2)
            # 旋转 返回 center(x,y), (w,h), angle, angle定义: 以旋转矩形框y最小,y相等时x最小的点为旋转点，以x轴正方向开始顺时针旋转碰到旋转矩形框第一条边时所转过的角度，
            # 旋转重合的第一条边算做width,另一条算做height,其取值范围为 ( 0 ,π/2] ,当x轴与一条边重合时取 π / 2 
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.intp(box)

            if rect[2]<=45:
                width = rect[1][0]
                height = rect[1][1]
                angle = rect[2] + 90
            else:
                width = rect[1][1]
                height = rect[1][0]
                angle = rect[2]

            wrinkle_list.append( (rect[0], width, height, angle, box, [x,y], [x+w,y+h]) )


            cv.drawContours(image,[box],0,(0,0,255),2)
            cnt += 1
    return image, wrinkle_list
# 结合img和人工标注的json处理
def main(image_path, json_path, ):
    img = cv.imread(image_path)
    with open(json_path) as f:
        labels = json.load(f)
    result = []
    cnt = 0
    for label in labels['shapes']:
        h1 = label['points'][0][1]
        h2 = label['points'][1][1]
        if h1>h2:
            tmp = h2
            h2 = h1
            h1 = tmp
        w1 = label['points'][0][0]
        w2 = label['points'][1][0]
        if w1>w2:
            tmp = w2
            w2 = w1
            w1 = tmp
        area = (w2-w1)*(h2-h1)
        crop = img[ int(h1): int(h2) , int(w1): int(w2) ]
        # cv.imwrite('crop.jpg', crop)
        binary = process_image(crop)
        # cv.imwrite('bw.jpg', binary)

        contour, max_w = drawContours(binary)
        # cv.imwrite('draw'+ str(cnt)  +'.jpg', contour)

        hough_, avg_deg = hough(binary,100, int(crop.shape[0]/3) ,30)
        # cv.imwrite('bihough'+ str(cnt) +'.jpg',hough_)

        # hough1 = hough(crop,100, int(crop.shape[0]/3) ,30)
        # cv.imwrite('hough'+ str(cnt) +'.jpg',hough1)
        # level, degree, num('multiple' or ''), loc ,area
        result.append( [ label['label'], avg_deg, label['description'], [(w1+w2)/2 , (h1+h2)/2] , area ,max_w] )
        cnt += 1
    return result

# 二值化后自动找皱褶，生成json
def main2(image_path, ):
    img = cv.imread(image_path)
    
    binary = process_image(img)
    # cv.imwrite('bw.jpg', binary)

    contour, wrinkles = drawContours(binary)
    # cv.imwrite('draw.jpg', contour)

    # TODO

    #目前用cv.findCounters再寻找最小旋转矩形也可以找到方向，暂时不需要houghline
    # hough_, avg_deg = hough(binary,100, int(img.shape[0]/3) ,30)
    # cv.imwrite('bihough.jpg',hough_)

    # hough1 = hough(crop,100, int(crop.shape[0]/3) ,30)
    # cv.imwrite('hough'+ str(cnt) +'.jpg',hough1)
    # level, degree, num('multiple' or ''), loc ,area
    return wrinkles

def to_json(input_path, output_path, wrinkles):
    # file_name, extension = os.path.splitext(input_path)
    file_name =  os.path.splitext(input_path)[0].split('/')[-1]
    # img = cv.imread(image_path)
        # 读取图像文件并进行Base64编码
    with open(input_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    labelme_data = {
    "version": "5.2.0.post4",
    "flags": {},
    "shapes": [],
    "imagePath": "path/to/your/image.jpg",
    "imageData": encoded_image,
    "imageHeight": 500,
    "imageWidth": 4096
    }
    for wrinkle in wrinkles:
        if wrinkle[3]>90:
            point = [
                [
                    wrinkle[6][0],
                    wrinkle[5][1]
                ],
                [
                    wrinkle[5][0],
                    wrinkle[6][1]
                ]
            ]
        else:
            point = [
                [
                    wrinkle[5][0],
                    wrinkle[5][1]
                ],
                [
                    wrinkle[6][0],
                    wrinkle[6][1]
                ]
            ]
        shape = {
            "label": "wrinkle",
            "points": point,
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags":{}
        }
        labelme_data['shapes'].append(shape)

    with open(os.path.join(output_path, file_name)+'.json', 'w') as f:
        json.dump(labelme_data, f, indent = 4)




if __name__ == '__main__':
    # 获取当前目录
    # root_path = os.getcwd()
    # 
    # 实验临时用目录 
    # root_path = '/home/sstl/hhm/datasets/wrinkle_data_test3/images/'
    root_path = '/home/sstl/Disk1/data/silicon/datasets/1/images'
    output_path = '/home/sstl/Disk1/data/silicon/datasets/1/json'
#  /opt/Data/HHM/data/wrinkle_data_train3

    # image_path = './2023-04-02(12.58)/3_Image97.jpg'
    # json_path = '../2023-04-02(12.58)/3_Image97.json'

    img_list = os.listdir(root_path)
    sorted_img_list = sorted(img_list, key=lambda x: datetime.strptime(x, '%H:%M:%S.jpg'))

    for item in sorted_img_list:
        if item == sorted_img_list[3057]:
            break
        print(item)
        # if item.rsplit('.')[0] in os.listdir(output_path) and os.listdir(output_path+item.rsplit('.')[0]) != []:
        if item in os.listdir(output_path):
            print(f'{item} already exist')
            continue
        else:
            wrinkles = main2(os.path.join(root_path, item))
            to_json(os.path.join(root_path, item), output_path, wrinkles)

    # #遍历当前目录文件、文件夹
    # for root, dirs, files in os.walk(root_path):
    #     # print("Current folder: {}".format(root))
    #     for file_name in files:
    #         full_path = os.path.join(root, file_name)
    #         if os.path.isdir(full_path):
    #             pass
    #         else:
    #             print("{}".format(file_name))
    #             if os.path.splitext(full_path)[-1] == '.jpg':
    #                 wrinkles = main2(full_path)
    #                 to_json(full_path, wrinkles)
    #
    # wrinkles = main2(image_path) #
    # print(wrinkles[0])
    # a = to_json(image_path,wrinkles)
    #
    #
    #
    #
    # # 创建SinglePassClustering对象
    # clustering = SinglePassClustering(threshold=0.5, l2 = 0.1,l3 = 0.1)
    #
    # # 添加数据点
    # for i in wrinkles:
    #
    #     i_ = [np.array(i[0]), i[1], i[2], i[3]]
    #     clustering.add_point(i_)
    #
    # # 获取聚类结果
    # clusters = clustering.get_clusters()

    # 打印聚类结果
    # for i, cluster in enumerate(clusters):
    #     print(f"Cluster {i+1}:")
    #     print("Centroid:", cluster.centroid)
    #     print("Members:", cluster.members)
    #     print()