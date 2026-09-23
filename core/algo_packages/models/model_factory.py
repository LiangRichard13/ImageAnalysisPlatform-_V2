import os
import torch
import numpy as np
from torch.optim import Adam
from models import predrnn, predgru, convlstm, predgru_v2
from torchinfo import summary

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predgru': predgru.RNN,
            'predgru_v2': predgru_v2.RNN,
            'predrnn': predrnn.RNN,
            'convlstm': convlstm.ConvLSTM
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)

    def h_grad(self, grad):
        return grad

    @torch.no_grad()
    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model'+'-' + itr + '.ckpt')
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        # self.network.load_state_dict(stats['net_param'])
        self.network.load_state_dict(stats['net_param'], strict=False)

    def train(self, frames, mask, itr):
        frames_tensor = torch.FloatTensor(np.array(frames)).to(self.configs.device)
        mask_tensor = torch.FloatTensor(np.array(mask)).to(self.configs.device)
        self.optimizer.zero_grad()
        _, loss = self.network(frames_tensor, mask_tensor)
        loss.backward()
        self.optimizer.step()

        # self.network.cell_list[3].conv_h[1].weight.register_hook(self.h_grad)
        
        # if itr % self.configs.display_interval == 0:
        # # grad = self.network.cell_list[3].conv_h[1].weight.grad
        #     grad = self.network.cell_list[3].conv_h[1].weight.grad  # [192, 25, 200]
        #     # TODO reshape and save

        #     patch_height = int(self.configs.img_height / self.configs.patch_size)
        #     patch_width = int(self.configs.img_width / self.configs.patch_size)
        #     # temp = img.reshape(self.configs.batch_size, self.configs.input_length, patch_height, patch_width, self.configs.patch_size, self.configs.patch_size, self.configs.img_channel)
        #     temp = grad.reshape(self.configs.batch_size, 1, patch_height, patch_width, self.configs.patch_size, self.configs.patch_size, self.configs.img_channel)
        #     temp = temp.transpose(4, 3)
        #     # image = temp.reshape(self.configs.batch_size, self.configs.input_length, self.configs.img_height, self.configs.img_width, self.configs.img_channel)
        #     image = temp.reshape(self.configs.batch_size, 1, self.configs.img_height, self.configs.img_width, self.configs.img_channel).squeeze(-1)

        return loss.detach().cpu().numpy()

    @torch.no_grad()
    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(np.array(frames)).to(self.configs.device)
        mask_tensor = torch.FloatTensor(np.array(mask)).to(self.configs.device)

        # summary(self.network, input_data=(frames_tensor, mask_tensor))

        next_frames, _ = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()
    