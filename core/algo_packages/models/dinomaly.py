







import torch
import torch.nn as nn
from models.uad import ViTill, ViTillv2
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2
from functools import partial
from dinov1.utils import trunc_normal_



class Dinomaly(nn.Module):

    def __init__(self):
        super(Dinomaly, self).__init__()
        encoder_name = 'dinov2reg_vit_base_14'
        # encoder_name = 'dinov2reg_vit_large_14'

        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        # target_layers = list(range(4, 19))

        encoder = vit_encoder.load(encoder_name)

        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."

        bottleneck = []
        decoder = []

        bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
        bottleneck = nn.ModuleList(bottleneck)

        for i in range(8):
            blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                        attn=LinearAttention2)
            decoder.append(blk)
        decoder = nn.ModuleList(decoder)

        model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                    mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
        model = model.to(0)
        trainable = nn.ModuleList([bottleneck, decoder, model.prompt])

        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.model = model
        self.trainable = trainable
        # self.execution_order = []
        # def forward_hook(module, input, output):
        #     if hasattr(module, 'weight'):
        #         self.execution_order.append(module)

        # # 在所有子模块上注册前向钩子
        # for module in model.modules():
        #     if isinstance(module, nn.Linear):  # 根据需要选择要监控的层类型
        #         module.register_forward_hook(forward_hook)
        # dummy_input = torch.randn([16, 3, 392, 392]).cuda()

        # # 执行前向传播
        # model(dummy_input)
        # print(self.execution_order)
        # for n, p in self.model.named_parameters():
        #     print(n)

    def forward(self, x):
        return self.model(x)