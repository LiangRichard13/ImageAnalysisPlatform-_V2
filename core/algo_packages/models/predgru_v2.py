import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SpatioTemporalGRUCell import SpatioTemporalGRUCell
from a100_utils.preprocess import reshape_patch, reshape_patch_back
from preprocess.tojson import process_image, drawContours
from a100_utils.eval_bbox import IoU, distance_square, diagonal_square
import cv2
import numpy as np

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * \
            configs.patch_size * configs.img_channel
        # self.configs.patch_size
        
        self.num_layers = num_layers  # self.num_layers = len(self.num_hidden)
        self.num_hidden = num_hidden
        
        cell_list = []

        self.h_t = []

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
        self.tconv = nn.ConvTranspose2d(self.frame_channel, num_hidden[0], 1, 1, padding=0, bias=False)
        # self.conv1 = nn.Conv2d(num_hidden[0] * 2, num_hidden[0], kernel_size=1)


    # def h_grad(self, grad):
    #     return grad


    def area(self, box):
        width = torch.max(box[2] - box[0], 0).values
        height = torch.max(box[3] - box[1], 0).values
        return width * height

    # def bbox(self, img, threshold):
    #     # TODO: tensor version
    #     # temp = reshape_patch_back(h_init.permute(0, 1, 3, 4, 2).cpu().numpy(), self.configs.patch_size)  # (1, 30, 100, 800, 1)
    #     # temp = torch.asarray(temp).to(self.configs.device)
    #     img_reshape = reshape_patch_back(img.cpu().detach().numpy(), self.configs.patch_size)  # (1, 29, 100, 800, 1)
    #     imgs = img_reshape.squeeze()  # (29, 100, 800)

    #     Bboxs = []
    #     for i in range(imgs.shape[0]):
    #         binary = process_image(cv2.Mat(img_reshape.squeeze()[i]).astype(np.uint8) * 255)
    #         # img_bi = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    #         _, wrinkles = drawContours(binary)
    #         X_min, Y_min, X_max, Y_max = [800, 100, 0, 0]
    #         for wrinkle in wrinkles:
    #             x1 = wrinkle[5][0]
    #             y1 = wrinkle[5][1]
    #             x2 = wrinkle[6][0]
    #             y2 = wrinkle[6][1]
    #             if max(x1, x2) > X_max:
    #                 X_max = max(x1, x2)
    #             if min(x1, x2) < X_min:
    #                 X_min = min(x1, x2)
    #             if max(y1, y2) > Y_max:
    #                 Y_max = max(y1, y2)
    #             if min(y1, y2) < Y_min:
    #                 Y_min = min(y1, y2)
    #         Bbox = [X_min, Y_min, X_max, Y_max]
    #         Bboxs.append(Bbox)
    #     return Bboxs
    
    def new_bbox(self, img):
        
        # TODO input multiple imgs

        # 定义参数
        # low_threshold = 100 / 255
        # high_threshold = 150 / 255
        dilation_size = 3
        # block_size = 11
        # C = 2

        patch_height = int(self.configs.img_height / self.configs.patch_size)
        patch_width = int(self.configs.img_width / self.configs.patch_size)
        # temp = img.reshape(self.configs.batch_size, self.configs.input_length, patch_height, patch_width, self.configs.patch_size, self.configs.patch_size, self.configs.img_channel)
        temp = img.reshape(self.configs.batch_size, 1, patch_height, patch_width, self.configs.patch_size, self.configs.patch_size, self.configs.img_channel)
        temp = temp.transpose(4, 3)
        # image = temp.reshape(self.configs.batch_size, self.configs.input_length, self.configs.img_height, self.configs.img_width, self.configs.img_channel)
        image = temp.reshape(self.configs.batch_size, 1, self.configs.img_height, self.configs.img_width, self.configs.img_channel).squeeze(-1)

        # image = img.flatten().unflatten(0, (self.configs.batch_size, 1, self.configs.img_height, self.configs.img_width))


        # Canny 边缘检测的近似实现
        # Sobel卷积核用于计算梯度
        # sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.configs.device).view(1, 3, 3)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.configs.device).view(1, 1, 3, 3)
        # sobel_x = torch.stack([sobel_x] * self.configs.input_length, dim=0)
        # sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.configs.device).view(1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.configs.device).view(1, 1, 3, 3)
        # sobel_y = torch.stack([sobel_y] * self.configs.input_length, dim=0)

        # sobel_1 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32, device=self.configs.device).view(1, 1, 3, 3)
        # sobel_2 = torch.tensor([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=torch.float32, device=self.configs.device).view(1, 1, 3, 3)
        # gradient_1 = F.conv2d(image, sobel_1, padding=0)
        # gradient_2 = F.conv2d(image, sobel_2, padding=0)

        # 计算图像的梯度
        # gradient_x = F.conv2d(image.squeeze(-1), sobel_x, padding=1, groups=self.configs.input_length)
        gradient_x = F.conv2d(image, sobel_x, padding=0)
        # gradient_y = F.conv2d(image.squeeze(-1), sobel_y, padding=1, groups=self.configs.input_length)
        gradient_y = F.conv2d(image, sobel_y, padding=0)
        # 计算梯度的幅度和方向
        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        # gradient_direction = torch.atan2(gradient_y, gradient_x)

        # gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + gradient_1 ** 2 + gradient_2 ** 2)


        # s_1 = torch.tensor([[0, 0, 0, 0, 0], [-1, -2, -4, -2, -1], [0, 0, 0, 0, 0], [1, 2, 4, 2, 1], [0, 0, 0, 0, 0]], dtype=torch.float32, device=self.configs.device).view(1, 1, 5, 5)
        # s_2 = torch.tensor([[0, 0, 0, 0, 0], [0, -2, -4, -2, 0], [-1, -4, 0, 4, 1], [0, 2, 4, 2, 0], [0, 0, 0, 0, 0]], dtype=torch.float32, device=self.configs.device).view(1, 1, 5, 5)
        # s_3 = torch.tensor([[0, 0, 0, -1, 0], [0, -2, -4, 0, 1], [0, -4, 0, 4, 0], [-1, 0, 4, 2, 0], [0, 1, 0, 0, 0]], dtype=torch.float32, device=self.configs.device).view(1, 1, 5, 5)
        # s_4 = torch.tensor([[0, 0, -1, 0, 0], [0, -2, -4, 2, 0], [0, -4, 0, 4, 0], [0, -2, 4, 2, 0], [0, 0, 1, 0, 0]], dtype=torch.float32, device=self.configs.device).view(1, 1, 5, 5)
        # s_5 = torch.tensor([[0, -1, 0, 1, 0], [0, -2, 0, -2, 0], [0, -4, 0, 4, 0], [0, -2, 0, 2, 0], [0, -1, 0, 1, 0]], dtype=torch.float32, device=self.configs.device).view(1, 1, 5, 5)
        # s_6 = torch.tensor([[0, 1, 0, 0, 0], [0, -2, 4, 2, 0], [0, -4, 0, 4, 0], [0, 2, -4, 2, 0], [0, 0, -1, 0, 0]], dtype=torch.float32, device=self.configs.device).view(1, 1, 5, 5)
        # s_7 = torch.tensor([[0, 1, 0, 0, 0], [-1, 0, 4, 2, 0], [0, -4, 0, 4, 0], [0, -2, -4, 0, 1], [0, 0, 0, -1, 0]], dtype=torch.float32, device=self.configs.device).view(1, 1, 5, 5)
        # s_8 = torch.tensor([[0, 0, 0, 0, 0], [0, 2, 4, 2, 0], [-1, -4, 0, 4, 1], [0, -2, -4, -2, 0], [0, 0, 0, 0, 0]], dtype=torch.float32, device=self.configs.device).view(1, 1, 5, 5)

        # g_1 = F.conv2d(image, s_1, padding=0)
        # g_2 = F.conv2d(image, s_2, padding=0)
        # g_3 = F.conv2d(image, s_3, padding=0)
        # g_4 = F.conv2d(image, s_4, padding=0)
        # g_5 = F.conv2d(image, s_5, padding=0)
        # g_6 = F.conv2d(image, s_6, padding=0)
        # g_7 = F.conv2d(image, s_7, padding=0)
        # g_8 = F.conv2d(image, s_8, padding=0)

        # gradient_magnitude = torch.sqrt(g_1 ** 2 + g_2 ** 2 + g_3 ** 2 + g_4 ** 2 + g_5 ** 2 + g_6 ** 2 + g_7 ** 2 + g_8 ** 2)

        greater = (gradient_magnitude - 1).relu()

        # 沿着列方向应用any函数，得到每列是否至少有一个值
        col_greater = greater.any(dim=2).float()

        col_greater = torch.stack([col_greater] * 100, dim=2)

        # 为了计算梯度，我们需要一个可导的操作，这里我们创建一个自定义的标量，并将其与结果相乘
        # 这个标量是可导的，因此整个操作是可导的
        scalar = torch.tensor(1.0, requires_grad=True)

        # 将结果与标量相乘
        mask = col_greater * scalar

        """visulize"""
        # cv2.imwrite('test.jpg', image.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('gx.jpg', gradient_x.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('gy.jpg', gradient_y.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('grad.jpg', gradient_magnitude.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('grad_direction.jpg', gradient_direction.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('filt.jpg', greater.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('mask.jpg', mask.squeeze().cpu().detach().numpy()*255)

        return mask


    def new_bbox_multi(self, imgs):
        patch_height = int(self.configs.img_height / self.configs.patch_size)
        patch_width = int(self.configs.img_width / self.configs.patch_size)
        temp = imgs.reshape(self.configs.batch_size, self.configs.input_length, patch_height, patch_width, self.configs.patch_size, self.configs.patch_size, self.configs.img_channel)
        temp = temp.transpose(4, 3)
        image = temp.reshape(self.configs.batch_size, self.configs.input_length, self.configs.img_height, self.configs.img_width, self.configs.img_channel).squeeze(-1)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.configs.device).view(1, 3, 3)
        sobel_x = torch.stack([sobel_x] * self.configs.input_length, dim=0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.configs.device).view(1, 3, 3)
        sobel_y = torch.stack([sobel_y] * self.configs.input_length, dim=0)

        gradient_x = F.conv2d(image.squeeze(-1), sobel_x, padding=1, groups=self.configs.input_length)
        gradient_y = F.conv2d(image.squeeze(-1), sobel_y, padding=1, groups=self.configs.input_length)

        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_direction = torch.atan2(gradient_y, gradient_x)


        greater = (gradient_magnitude - 1).relu()

        col_greater = greater.any(dim=2).float()
        col_greater = torch.stack([col_greater] * 100, dim=2)
        scalar = torch.tensor(1.0, requires_grad=True)

        mask = col_greater * scalar

        """visulize"""
        # cv2.imwrite('test.jpg', image.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('gx.jpg', gradient_x.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('gy.jpg', gradient_y.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('grad.jpg', gradient_magnitude.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('grad_direction.jpg', gradient_direction.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('filt.jpg', greater.squeeze().cpu().detach().numpy()*255)
        # cv2.imwrite('mask.jpg', mask.squeeze().cpu().detach().numpy()*255)

        return mask


    def new_iou(self, frames_tensor, next_frames):
        gt_bbox = self.new_bbox(frames_tensor)
        pd_bbox = self.new_bbox(next_frames)

        # gt_bbox = self.new_bbox_multi(frames_tensor)
        # pd_bbox = self.new_bbox_multi(next_frames)

        inter = torch.sum(gt_bbox * pd_bbox)
        union = torch.sum(torch.maximum(gt_bbox, pd_bbox))
        return inter / union


    def iou(self, frames_tensor, next_frames):

        # img_path = os.path.join(new_path, i)
        # img = cv2.imread(img_path)
        # threshold_gt = 10
        threshold_gt = 2
        threshold_pred = 2  # wrinkle2
        # threshold_pred = 4  # wrinkle3
                               
        bbox_gt = self.bbox(frames_tensor[0][:, 1:], threshold_gt)
        bbox_pred = self.bbox(next_frames, threshold_pred)
        ious = []
        for i in range(len(bbox_gt)):
            # cv2.imwrite('pred.png', cv2.Mat(reshape_patch_back(next_frames.cpu().detach().numpy(), self.configs.patch_size).squeeze()[i].astype(np.uint8))*255)
            # cv2.imwrite('gt.png', cv2.Mat(reshape_patch_back(frames_tensor[0][:, 1:].cpu().detach().numpy(), self.configs.patch_size).squeeze()[i].astype(np.uint8))*255)
            iou = IoU(bbox_gt[i], bbox_pred[i])
            # iou = PRO(bbox_gt[i], bbox_pred[i])
            ious.append(iou)
            # pros.append(pro)
        ious = torch.Tensor(ious)
        return torch.mean(ious)
    
    def giou(self, frames_tensor, next_frames):
        return 0
    
    def diou(self, frames_tensor, next_frames):
        threshold_gt = 2
        threshold_pred = 2  # wrinkle2
        # threshold_pred = 4  # wrinkle3
                               
        bbox_gt = self.bbox(frames_tensor[0][:, 1:], threshold_gt)
        bbox_pred = self.bbox(next_frames, threshold_pred)
        dious = []
        for i in range(len(bbox_gt)):
            iou = IoU(bbox_gt[i], bbox_pred[i])
            diou = iou - distance_square(bbox_gt[i], bbox_pred[i]) / diagonal_square(bbox_gt[i], bbox_pred[i])
            dious.append(diou)

        dious = torch.Tensor(dious)
        return torch.mean(dious)


    def weighted_loss(self, masks, inputs, targets): 
        # diff = (inputs - targets).permute(1, 0, 4, 2, 3)
        diff = inputs - targets
        # weighted_diff = 0.01 * masks * diff
        # diff_square = weighted_diff ** 2
        # return diff_square.mean()
        
        diff_square = diff ** 2
        # return masks * diff_square.mean()
        return (masks * diff_square).mean()
        
    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor[0].permute(0, 1, 4, 2, 3).contiguous()
        h_init = frames_tensor[1].permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        self.h_t = []
        # c_t = []
        # plot
        # matrices_x = []
        # matrices_h = []
        # # matrices_c = []
        # matrices_m = []
        
        temp = reshape_patch_back(h_init.permute(0, 1, 3, 4, 2).cpu().numpy(), self.configs.patch_size)  # (1, 30, 100, 800, 1)
        temp = torch.asarray(temp).to(self.configs.device)

        res = []
        for i in range(len(temp.squeeze())):
            max_values, _ = torch.max(temp.squeeze()[i], dim=0)
            # max_values = torch.mean(temp.squeeze()[i], dim=0)
            # TODO: value resacle
            restore = max_values.unsqueeze(1).expand(-1, 100)
            res.append(restore)
            
        max = torch.stack(res).unsqueeze(0).unsqueeze(-1)
        max_patch = reshape_patch(max.cpu().numpy(), self.configs.patch_size)  # (1, 30, 25, 200, 16)
        max_patch = torch.asarray(max_patch).to(self.configs.device).permute(0, 1, 3, 2, 4)
        
        # reshape_patch(reshape_patch_back(h_init.permute(0, 1, 3, 4, 2).cpu().numpy(), self.configs.patch_size), self.configs.patch_size)  # (1, 30, 25, 200, 16)

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
            # schedule sampling
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
            # h_t[0], memory = self.cell_list[0](net, h_t[0], memory)
            # 2
            # h_t[0] = self.cell_list[0](net, h_t[0])
            # 3
            mask_tconv = self.tconv(mask)
            
            if t == 0:
                for i in range(self.num_layers):
                     self.h_t.append(mask_tconv)
                self.h_t[0], memory = self.cell_list[0](net, self.h_t[0], memory)
            else:
                self.h_t[0], memory = self.cell_list[0](net, self.h_t[0] * mask_tconv, memory)
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
                # h_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], memory)
                # 2
                # h_t[i] = self.cell_list[i](h_t[i - 1], h_t[i])
                # 3
                # h_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i] * mask_tconv, memory)
                if t == 0:
                    self.h_t[i], memory = self.cell_list[i](self.h_t[i - 1], self.h_t[i], memory)
                else:
                    self.h_t[i], memory = self.cell_list[i](self.h_t[i - 1], self.h_t[i] * mask_tconv, memory)
                    # concat = torch.cat((h_t[i], mask_tconv), dim=1)
                    # h_conv1 = self.conv1(concat)
                    # h_t[i], memory = self.cell_list[i](h_t[i - 1], h_conv1, memory)

            # TODO: save h grad
            # self.h_t[3].register_hook(self.h_grad)

            x_gen = self.conv_last(self.h_t[self.num_layers - 1])  # h_t=3,
            # mask_weight = self.conv_last(mask_tconv)  # uniform the channel of mask and frame
            # mask_normalized = (mask_weight - torch.min(mask_weight)) / (torch.max(mask_weight) - torch.min(mask_weight))  # min-max normalization
            # masks.append(mask_normalized)
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        # masks_tensor = torch.stack(masks)
        
        # loss = self.MSE_criterion(next_frames, frames_tensor[0][:, 1:])  # 忽略第一帧

        # loss = self.weighted_loss(max_patch[:, 1:], next_frames, frames_tensor[0][:, 1:]) + 0.001 * (1 - self.iou(frames_tensor, next_frames))
        # loss = self.MSE_criterion(next_frames, frames_tensor[0][:, 1:]) + 0.001 * (1 - self.iou(frames_tensor, next_frames))
        
        # loss = self.MSE_criterion(next_frames, frames_tensor[0][:, 1:])
        # loss = self.weighted_loss(max_patch[:, 1:], next_frames, frames_tensor[0][:, 1:])


        # mse_loss = self.MSE_criterion(next_frames, frames_tensor[0][:, 1:])
        mse_loss = self.weighted_loss(max_patch[:, 1:], next_frames, frames_tensor[0][:, 1:])
        iou_loss = 1 - self.new_iou(frames_tensor[0][:, -1:, :, :, :], next_frames[:, -1:, :, :, :])
        lambda1 = 1
        lambda2 = 0.001
        # lambda2 = 0.005
        loss = lambda1 * mse_loss + lambda2 * iou_loss

        #multi
        # loss = 1 - self.new_iou(frames_tensor[0][:, 1:], next_frames)
        # single
        # loss = 1 - self.new_iou(frames_tensor[0][:, -1:, :, :, :], next_frames[:, -1:, :, :, :])

        # loss = 1 - self.diou(frames_tensor[0][:, 1:], next_frames)
        return next_frames, loss
