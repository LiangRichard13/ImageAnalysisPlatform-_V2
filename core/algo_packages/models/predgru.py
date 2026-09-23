import torch
import torch.nn as nn
from layers.SpatioTemporalGRUCell import SpatioTemporalGRUCell


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

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
            # 如果是第一个cell，iputchannel=图片通道数，如果不是，就等于上一层的通道数
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalGRUCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                                      configs.stride, configs.layer_norm)
            )
        # 将这个列表转化为模块列表。这样做的目的是让PyTorch知道cell_list中所有的SpatioTemporalLSTMCell都是网络的一部分，从而在计算梯度和更新权重时能够正确地处理这些SpatioTemporalLSTMCell。
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)  # 输出转为原始图像的通道数，方便比较和作为下一个时间步输入
        # 3
        # self.tconv = nn.ConvTranspose2d(self.frame_channel, num_hidden[0], 1, 1, padding=0, bias=False)
        
        # self.conv1 = nn.Conv2d(num_hidden[0] * 2, num_hidden[0], kernel_size=1)

    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor[0].permute(0, 1, 4, 2, 3).contiguous()
        h_init = frames_tensor[1].permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        # c_t = []
        # plot
        # matrices_x = []
        # matrices_h = []
        # # matrices_c = []
        # matrices_m = []

        # ..
        # for i in range(self.num_layers):
        #     # temp = h_init[:, 0+i*4 : 4+i*4, :, :, :]  # torch.Size([1, 4, 16, 25, 200])
        #     # temp_reshape = torch.flatten(temp, start_dim=1, end_dim=2)  # torch.Size([1, 64, 25, 200])
        #     # h_t.append(temp_reshape)
        #     # ones = torch.ones([batch, self.num_hidden[i], height, width]).to(self.configs.device)
        #     zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
        #     h_t.append(zeros)
            
        for i in range(self.num_layers):#初始化列表
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            # c_t.append(zeros)

        # 1
        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(
            self.configs.device)  # 初始化

        for t in range(self.configs.total_length - 1):  # 16-1=15,t取值0-14。。。。。
            # # reverse schedule sampling
            # if self.configs.reverse_scheduled_sampling == 1:
            #     if t == 0:
            #         net = frames[:, t]
            #     else:
            #         net = mask_true[:, t - 1] * frames[:, t] + \
            #             (1 - mask_true[:, t - 1]) * x_gen
            # # schedule sampling
            # else:
            if t < self.configs.input_length:  # <15，一直在这个循环
                net = frames[:, t]
                mask = h_init[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                    (1 - mask_true[:, t -
                        self.configs.input_length]) * x_gen

            # plot
            # matrices_x.append(net)
            # matrices_h.append(h_t[0])
            # # matrices_c.append(c_t[0])
            # matrices_m.append(memory)

            # h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            # 1
            h_t[0], memory = self.cell_list[0](net, h_t[0], memory)
            # 2
            # h_t[0] = self.cell_list[0](net, h_t[0])
            # 3

            # mask_tconv = self.tconv(mask)

            # if t == 0:
            #     for i in range(self.num_layers):
            #          h_t.append(mask_tconv)
            #     h_t[0], memory = self.cell_list[0](net, h_t[0], memory)
            # else:
            #     h_t[0], memory = self.cell_list[0](net, h_t[0] * mask_tconv, memory)

                # concat = torch.cat((h_t[0], mask_tconv), dim=1)
                # h_conv1 = self.conv1(concat)
                # h_t[0], memory = self.cell_list[0](net, h_conv1, memory)

            # print(net.shape)
            for i in range(1, self.num_layers):  # 1 2 3
                # plot
                # matrices_x.append(h_t[i - 1])
                # matrices_h.append(h_t[i])
                # # matrices_c.append(c_t[i])
                # matrices_m.append(memory)

                # h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                # 1
                h_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], memory)
                # 2
                # h_t[i] = self.cell_list[i](h_t[i - 1], h_t[i])
                # 3
                # h_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i] * mask_tconv, memory)


                # if t == 0:
                #     h_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], memory)
                # else:
                #     h_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i] * mask_tconv, memory)

                #     # concat = torch.cat((h_t[i], mask_tconv), dim=1)
                #     # h_conv1 = self.conv1(concat)
                #     # h_t[i], memory = self.cell_list[i](h_t[i - 1], h_conv1, memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])  # h_t=3,
            next_frames.append(x_gen)

        # plot
        # matrices_x = torch.stack(matrices_x).cpu()
        # matrices_h = torch.stack(matrices_h).cpu()
        # # matrices_c = torch.stack(matrices_c).cpu()
        # matrices_m = torch.stack(matrices_m).cpu()

        # np.save('matrices_x.npy', matrices_x.detach().numpy())
        # np.save('matrices_h.npy', matrices_h.detach().numpy())
        # # np.save('matrices_c.npy', matrices_c.detach().numpy())
        # np.save('matrices_m.npy', matrices_m.detach().numpy())

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        # TODO: add another loss, use yolov5 to detect next_frames and frames[0], calculate the IoU and mutiply a balance coeficient
        # TODO: check frames_tensor[0][:, 1:]
        loss = self.MSE_criterion(next_frames, frames_tensor[0][:, 1:])  # 忽略第一帧
        return next_frames, loss
