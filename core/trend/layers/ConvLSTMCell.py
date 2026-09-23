import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()
        
        self.in_channel = in_channel
        self.num_hidden = num_hidden
        self.height = height
        self.width = width
        self.filter_size = filter_size
        self.stride = stride
        self.layer_norm = layer_norm
        
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        
        # ST-LSTM: 7 4 3 2 
        # GRU: 6 3 3
        
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(self.in_channel, self.num_hidden * 4, kernel_size=self.filter_size, stride=self.stride, padding=self.padding, bias=False),
                nn.LayerNorm([self.num_hidden * 4, self.height, self.width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(self.num_hidden, self.num_hidden * 4, kernel_size=self.filter_size, stride=self.stride, padding=self.padding, bias=False),
                nn.LayerNorm([self.num_hidden * 4, self.height, self.width])
            )

        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(self.in_channel, self.num_hidden * 4, kernel_size=self.filter_size, stride=self.stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(self.num_hidden, self.num_hidden * 4, kernel_size=self.filter_size, stride=self.stride, padding=self.padding, bias=False),
            )

        # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=4 * self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t) #当前步x，前一个时间布h，上一层m
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        o_t = torch.sigmoid(o_x + o_h)
        
        c_new = f_t * c_t + i_t * g_t
        h_new = o_t * torch.tanh(c_new)
        
        return h_new, c_new
        

        
        

    # def forward(self, input_tensor, cur_state):
    #     h_cur, c_cur = cur_state

    #     combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

    #     combined_conv = self.conv(combined)
    #     cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
    #     i = torch.sigmoid(cc_i)
    #     f = torch.sigmoid(cc_f)
    #     o = torch.sigmoid(cc_o)
    #     g = torch.tanh(cc_g)

    #     c_next = f * c_cur + i * g
    #     h_next = o * torch.tanh(c_next)

    #     return h_next, c_next

    # def init_hidden(self, batch_size, image_size):
    #     height, width = image_size
    #     return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
    #             torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))