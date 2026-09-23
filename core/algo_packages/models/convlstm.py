import torch
import torch.nn as nn
from layers.ConvLSTMCell import ConvLSTMCell
import numpy as np


class ConvLSTM(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(ConvLSTM, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * \
            configs.patch_size * configs.img_channel
        self.num_layers = num_layers  # self.num_layers = len(self.num_hidden)
        self.num_hidden = num_hidden
        
        cell_list = []

        height = configs.img_height // configs.patch_size
        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                                      configs.stride, configs.layer_norm)
            )
        # 将这个列表转化为模块列表。这样做的目的是让PyTorch知道cell_list中所有的SpatioTemporalLSTMCell都是网络的一部分，从而在计算梯度和更新权重时能够正确地处理这些SpatioTemporalLSTMCell。
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)  # 输出转为原始图像的通道数，方便比较和作为下一个时间步输入   
        

    def forward(self, frames_tensor, mask_true):
        frames = frames_tensor[0].permute(0, 1, 4, 2, 3).contiguous()
        # h_init = frames_tensor[1].permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        
        for i in range(self.num_layers):#初始化列表
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        # memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)#初始化

        for t in range(self.configs.total_length - 1):#16-1=15,t取值0-14。。。。。
            # # reverse schedule sampling
            # if self.configs.reverse_scheduled_sampling == 1:
            #     if t == 0:
            #         net = frames[:, t]
            #     else:
            #         net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            # # schedule sampling
            # else:
            if t < self.configs.input_length: #<15，一直在这个循环
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                        (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            
            # h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])
            
            # # print(net.shape)
            for i in range(1, self.num_layers):#1 2 3 
                # h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)     
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])             
            
            x_gen = self.conv_last(h_t[self.num_layers - 1])#h_t=3,
            next_frames.append(x_gen)


        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        # loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])#忽略第一帧
        loss = self.MSE_criterion(next_frames, frames_tensor[0][:, 1:])  # 忽略第一帧
        return next_frames, loss
