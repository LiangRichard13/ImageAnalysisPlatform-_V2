from datetime import datetime
import numpy as np
import os
import cv2
# from PIL import Image
# import logging
import random
# import re

# logger = logging.getLogger(__name__)

def adjust_gamma(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    hist_eq = clahe.apply(image)
    return hist_eq


class InputHandle:
    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.image_height = input_param['image_height'] 
        # self.channel = input_param['channel']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']
        self.interval = 1

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        # logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size > self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            # logger.error(
            #     "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        # input_batch = np.zeros(
        #     (self.minibatch_size, self.current_input_length, self.image_height, self.image_width, 1)).astype(
        #     self.input_data_type)
        frame_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_height, self.image_width, 1)).astype(
            self.input_data_type)
        masks_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_height, self.image_width, 1)).astype(
            self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length * self.interval
            # data_slice = self.datas[:, begin:end:self.interval]
            # data_slice = np.expand_dims(data_slice, axis=-1)
            # # data_slice = self.datas[begin:end, :, :, :]
            # input_batch[i, :self.current_input_length, :, :, :] = data_slice
            frame_slice = self.datas[0][begin:end:self.interval]
            frame_slice = np.expand_dims(frame_slice, axis=-1)
            frame_batch[i, :self.current_input_length, :, :, :] = frame_slice
            masks_slice = self.datas[1][begin:end:self.interval]
            masks_slice = np.expand_dims(masks_slice, axis=-1)
            masks_batch[i, :self.current_input_length, :, :, :] = masks_slice
            frame_batch = frame_batch.astype(self.input_data_type)
            masks_batch = masks_batch.astype(self.input_data_type)
        return [frame_batch, masks_batch]

    # def print_stat(self):
    #     logger.info("Iterator Name: " + self.name)
    #     logger.info("    current_position: " + str(self.current_position))
    #     logger.info("    Minibatch Size: " + str(self.minibatch_size))
    #     logger.info("    total Size: " + str(self.total()))
    #     logger.info("    current_input_length: " + str(self.current_input_length))
    #     logger.info("    Input Data Type: " + str(self.input_data_type))


class DataProcess:
    def __init__(self, input_param):
        self.input_param = input_param
        self.paths = input_param['paths']
        self.image_width = input_param['image_width']
        self.image_height = input_param['image_height']
        self.seq_len = input_param['seq_length']


    def load_data(self, paths, mode='train'):
        # TODO: train 6 val 2 test 2

        data_dir = paths[0]
        mask_dir = paths[1]

        intervel = 1

        frames_np = []

        print('loading data from', data_dir)  
        filenames = os.listdir(data_dir)

        # new4
        # filenames=sorted(filenames, key=lambda x: datetime.strptime(x, '%H:%M:%S.jpg'))
        # 2/3
        filenames = sorted(filenames, key=lambda x: int(x.strip('Image').strip('.jpg')))

        # train 3057 * 0.7  = 2140
        # val 3057 * 0.1 = 305
        # test  3057 - 2140 - 306 = 612
        # test 3031 - 2140 - 306 = 585

        # total 928 + 272 = 1200
        # train 838 - 69.8%
        # val 90 - 7.49% missing one
        # test 272 - 22.6%

        # 926/262

        # TODO
        if mode == 'train':
            # filenames = filenames[:2140]
            filenames = filenames[:838]
        elif mode == 'val':
            # filenames = filenames[2140:2445]
            filenames = filenames[838:]
        elif mode == 'test':
            # filenames = filenames[2445:]
            filenames = filenames
        else:
            print("MODE ERROR")
            
        print(mode + ' data size ', len(filenames))  

        mask = np.load(mask_dir)
        masks_np = []

        for filename in filenames:  # filenames=['1_Image0.jpg', ]
            file_path = os.path.join(data_dir, filename)  # _path=/opt/Data/HHM/data/wrinkle_data_train2/
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            frames_np.append(np.array(image, dtype=np.float32) / 255.0)
            masks_np.append(mask[filename].astype(np.float32))

        indices = []
        index = 0
        # print('gen index')          
        while index + intervel * self.seq_len - 1 < len(filenames): 
            indices.append(index)
            index += 4

        print("there are " + str(len(indices)) + " sequences")
        # data = np.asarray(frames_np)
        data = frames_np, masks_np
        # print("there are " + str(len(frames_np)) + " pictures")
        return data, indices