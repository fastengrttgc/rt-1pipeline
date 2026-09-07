'''
2026/9/1 wangyuqi
测试模型依赖项完整性和数据链路
'''

import torch
from robotic_transformer_pytorch import MaxViT, RT1

vit = MaxViT(
    num_classes=1000, dim_conv_stem=64, dim=96, dim_head=32,
    depth=(2, 2, 5, 2), window_size=7,
    mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, dropout=0.1,
)

model = RT1(
    vit=vit, num_actions=11, depth=6, heads=8, dim_head=64,
    cond_drop_prob=0.2,
)

video = torch.randn(2, 3, 6, 224, 224)   # batch=2, RGB=3, 6帧, 224x224
instructions = ['bring me that apple sitting on the table', 'please pass the butter']

train_logits = model(video, instructions)
print('train_logits:', tuple(train_logits.shape))   # 预期 (2, 6, 11, 256)

model.eval()
eval_logits = model(video, instructions, cond_scale=3.)
print('eval_logits :', tuple(eval_logits.shape))    # 预期 (2, 6, 11, 256)