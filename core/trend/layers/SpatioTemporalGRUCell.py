import torch
import torch.nn as nn

class SpatioTemporalGRUCell(nn.Module):
    def __init__(self, in_channel, num_hidden,height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalGRUCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 6, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 6, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            # 1
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            
            
            # self.conv_o = nn.Sequential(
            #     nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            #     nn.LayerNorm([num_hidden, height, width])
            # )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 6, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            # 1
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            
            
            # self.conv_o = nn.Sequential(
            #     nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            # )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    # def forward(self, x_t, h_t, c_t, m_t):
    # 1
    def forward(self, x_t, h_t, m_t):
    # 2
    # def forward(self, x_t, h_t):
        x_concat = self.conv_x(x_t) #当前步x，前一个时间布h，上一层m
        h_concat = self.conv_h(h_t)
        # 1
        m_concat = self.conv_m(m_t)
        
        
        # i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        # i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        # i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        # 1
        r_x, z_x, g_x, r_x_prime, z_x_prime, g_x_prime = torch.split(x_concat, self.num_hidden, dim=1)
        # 2
        # r_x, z_x, g_x = torch.split(x_concat, self.num_hidden, dim=1)
        r_h, z_h, g_h = torch.split(h_concat, self.num_hidden, dim=1)
        # 1
        r_m, z_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)


        # i_t = torch.sigmoid(i_x + i_h)
        # f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        # g_t = torch.tanh(g_x + g_h)
        r_t = torch.sigmoid(r_x + r_h)
        z_t = torch.sigmoid(z_x + z_h)
        g_t = torch.tanh(g_x + r_t * g_h)
        
        # c_new = f_t * c_t + i_t * g_t

        # i_t_prime = torch.sigmoid(i_x_prime + i_m)
        # f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        # g_t_prime = torch.tanh(g_x_prime + g_m)
        
        # 1
        r_t_prime = torch.sigmoid(r_x_prime + r_m)
        z_t_prime = torch.sigmoid(z_x_prime + z_m)
        # g_t_prime = torch.tanh(g_x_prime + r_t_prime * g_m)
        g_t_prime = torch.tanh(g_x_prime + r_t_prime * g_m)

        # mem = torch.cat((c_new, m_new), 1)
        # o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        # h_new = o_t * torch.tanh(self.conv_last(mem)) #ot输出门，选择整合后的哪一部分输出

        # TODO!!!
        # m_new = (1 - z_t_prime) * m_t + r_t_prime * g_t_prime
        # h_new = (1 - z_t) * h_t + z_t * g_t + m_new

        m_new = z_t_prime * m_t + r_t_prime * g_t_prime
        h_new = (1 - z_t) * g_t + z_t * h_t

        # print(x_t.shape, h_t.shape, c_t.shape,m_t.shape)
        # print(i_x.shape, f_x.shape, g_x.shape,i_x_prime.shape)
        # print(i_h.shape, f_h.shape, g_h.shape,o_h.shape)
        # print(i_t.shape, f_t.shape, g_t.shape,c_new.shape,mem.shape,o_t.shape,h_new.shape)
        # print(x_concat.shape, h_concat.shape, m_concat.shape)
        # print(f_m.shape, g_m.shape, i_m.shape)
        
        # return h_new, c_new, m_new
        # 1

        return h_new, m_new
        # 2
        # return h_new

