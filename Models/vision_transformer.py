import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from torch.autograd import Variable

class Multiview_VIT(nn.Module):
    def __init__(self, args):
        super(Multiview_VIT, self).__init__()

        self.args = args
        in_channels = self.args.vit_in_channels  # 3
        num_classes = self.args.num_classes
        axial_img_size = self.args.axial_shape
        sagittal_img_size = self.args.sagittal_shape
        coronal_img_size = self.args.coronal_shape
        patch_size = self.args.patch_size  # 16
        emb_dim = self.args.vit_emb_dim  # 768
        depth = self.args.vit_depth  # 12
        num_heads = self.args.vit_num_heads  # 12
        dropout = self.args.dropout  # 0.1

        self.patch_embedding_axial = PatchEmbedding(in_channels, patch_size, emb_dim, axial_img_size)
        self.patch_embedding_sagittal = PatchEmbedding(in_channels, patch_size, emb_dim, sagittal_img_size)
        self.patch_embedding_coronal = PatchEmbedding(in_channels, patch_size, emb_dim, coronal_img_size)

        self.transformer = nn.Sequential(
            *[TransformerBlock(emb_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim*3, num_classes)


    def forward(self, x):

        # encode
        # x [B, 193, 229, 193]
        # x = x.unsqueeze(1)  # [B, 1, 193, 229, 193]

        # axial
        x_axial = x.clone()  # [B, 1, 193, 229, 193]
        x_axial = self.patch_embedding_axial(x_axial)
        emb_axial = self.transformer(x_axial)
        emb_axial = self.norm(emb_axial[:, 0])

        # sagittal
        x_sagittal = x.clone().permute(0, 1, 4, 3, 2)  # [B, 1, 193, 229, 193]
        x_sagittal = self.patch_embedding_sagittal(x_sagittal)  # [B, 256, 193]
        emb_sagittal = self.transformer(x_sagittal)
        emb_sagittal = self.norm(emb_sagittal[:, 0])

        # coronal
        x_coronal = x.clone().permute(0, 1, 2, 4, 3)  # [B, 1, 193, 193, 229]
        x_coronal = self.patch_embedding_coronal(x_coronal)  # [B, 256, 193]
        emb_coronal = self.transformer(x_coronal)
        emb_coronal = self.norm(emb_coronal[:, 0])

        emb_matrix = torch.cat([emb_axial.squeeze(-1), emb_sagittal.squeeze(-1), emb_coronal.squeeze(-1)], 1)

        output = self.classifier(emb_matrix)

        return output

class PatchEmbed_3d(nn.Module):
    """
    3D Volume to Patch Embedding
    """

    def __init__(self, img_size, patch_size, in_c, embed_dim, norm_layer=None):
        # embed_dim = 16*16*16 = 4096 token在flatten之后的长度
        super().__init__()
        # img_size = (img_size, img_size, img_size)
        # patch_size = (patch_size, patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        # 一共有多少个token(patche)  (224/16)*(224/16)*(224/16) = 14*14*14 = 2744

        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, P = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 输入图片的大小必须是固定的

        # x 大小224*224*224 经过k=16,s=16,c=4096的卷积核之后大小为 14*14*14*4096
        # flatten: [B, C, H, W, P] -> [B, C, HWP]   [B, 4096, 14, 14, 14] -> [B, 4096, 2744]
        # 对于Transfoemer模块，要求输入的是token序列，即 [num_token,token_dim] = [2744,4096]
        # transpose: [B, C, HWP] -> [B, HWP, C]   [B, 4096, 2744] -> [B, 2744, 4096]

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, img_size):
        super().__init__()
        # self.patch_size = patch_size
        self.patch_height, self.patch_width = patch_size
        self.emb_dim = emb_dim

        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size[0] // self.patch_height) * (img_size[1] // self.patch_width)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.randn(n_patches + 1, emb_dim))
        # self.pos_emb = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_emb
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.qkv = nn.Linear(emb_dim, emb_dim * 3, bias=False)
        self.fc_out = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dots = torch.einsum('bhid,bhjd->bhij', q, k) / self.head_dim ** 0.5
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.reshape(B, N, C)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.mlp(x))
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    输出：[num_token,token_dim] = [14*14,16*16*3]
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        # embed_dim = 16*16*3 = 768 token在flatten之后的长度
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 一共有多少个token(patche)  (224/16)*(224/16) = 14*14 = 196

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 输入图片的大小必须是固定的

        # x 大小224*224 经过k=16,s=16,c=768的卷积核之后大小为 14*14*768
        # flatten: [B, C, H, W] -> [B, C, HW]   [B, 768, 14, 14] -> [B, 768, 196]
        # 对于Transfoemer模块，要求输入的是token序列，即 [num_token,token_dim] = [14*14,16*16*3]
        # transpose: [B, C, HW] -> [B, HW, C]   [B, 768, 196] -> [B, 196, 768]

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

    class Attention(nn.Module):
        def __init__(self,
                     dim,  # 输入token的dim
                     num_heads=8,
                     qkv_bias=False,
                     qk_scale=None,
                     attn_drop_ratio=0.,
                     proj_drop_ratio=0.):
            super(Attention, self).__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop_ratio)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop_ratio)

        def forward(self, x):
            # [batch_size, num_patches + 1, total_embed_dim]
            B, N, C = x.shape

            # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
            # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
            # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head] 调整顺序
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
            # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
            attn = (q @ k.transpose(-2, -1)) * self.scale  # q dot-product k的转置，只对最后两个维度进行操作
            attn = attn.softmax(dim=-1)  # 对每一行进行softmax
            attn = self.attn_drop(attn)

            # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
            # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
            # reshape: -> [batch_size, num_patches + 1, total_embed_dim] 将多头的结果拼接在一起
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # print(x.shape)
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head] 调整顺序
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale # q dot-product k的转置，只对最后两个维度进行操作
        attn = attn.softmax(dim=-1) # 对每一行进行softmax
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim] 将多头的结果拼接在一起
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4., # 第一个全连接层节点个数是输入的四倍
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class VisionTransformer(nn.Module):
    def __init__(self, img_size, args):
        super().__init__()
        self.args = args
        in_channels = self.args.vit_in_channels #3
        num_classes = self.args.num_classes
        # img_size = self.args.  #224
        patch_size = self.args.patch_size_2d  #16
        emb_dim = self.args.vit_emb_dim  #768
        depth = self.args.vit_depth   # 12
        num_heads = self.args.vit_num_heads #12
        dropout = self.args.dropout #0.1

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.transformer = nn.Sequential(
            *[TransformerBlock(emb_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        # print(x.shape)
        x = self.norm(x[:, 0])
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

class VisionTransformer_stage(nn.Module):
    def __init__(self, args):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_stage, self).__init__()
        self.args = args
        in_c = self.args.vit_in_channels  # 3
        num_classes = self.args.num_classes
        img_size = self.args.Image_shape  #224
        patch_size = self.args.patch_size_3d  # 16
        embed_dim = self.args.vit_emb_dim  # 768
        depth = self.args.vit_depth  # 12
        num_heads = self.args.vit_num_heads  # 12
        drop_ratio = self.args.dropout  # 0.1

        mlp_ratio = 4.0
        qkv_bias = True
        qk_scale = None

        attn_drop_ratio = 0.
        drop_path_ratio = 0.
        embed_layer = PatchEmbed_3d
        norm_layer = None
        act_layer = None

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1  # num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # 默认参数
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # token/patch的个数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # parameter构建可训练参数，第一个1是batch size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # 位置编码的大小和加入分类token之后的大小相同
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 构建等差序列，dropout率是递增的
        #         self.blocks = nn.Sequential(*[
        #             Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                   drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #                   norm_layer=norm_layer, act_layer=act_layer)
        #             for i in range(depth)
        #         ])
        self.stage1 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[0], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[1], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[2], norm_layer=norm_layer, act_layer=act_layer))

        self.stage2 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[3], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[4], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[5], norm_layer=norm_layer, act_layer=act_layer))

        self.stage3 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[6], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[7], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[8], norm_layer=norm_layer, act_layer=act_layer))

        self.stage4 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[9], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[10], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[11], norm_layer=norm_layer, act_layer=act_layer))

        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        # print(x.shape)
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        # print(x.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 把cls_token复制batch_size份
        # print(cls_token.shape)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # print(x.shape)
        x = self.pos_drop(x + self.pos_embed)
        # print(x.shape)
        # x = self.blocks(x)
        x = self.stage1(x)
        # print(x.shape)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        # print(x.shape)
        x = self.head(x)  # 执行这里
        # print(x.shape)
        return x

if __name__ == "__main__":
    img = torch.randn(1, 1, 76, 94)
    vit = VisionTransformer(num_classes=3)
    preds = vit(img)
    print(preds.shape) 
