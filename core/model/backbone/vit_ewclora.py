import math
from functools import partial

import timm
import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp, PatchEmbed, trunc_normal_


class AttentionEWCLoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, rank=10):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lora_A_k = nn.Linear(dim, rank, bias=False)
        self.lora_B_k = nn.Linear(rank, dim, bias=False)
        self.lora_A_v = nn.Linear(dim, rank, bias=False)
        self.lora_B_v = nn.Linear(rank, dim, bias=False)

        self.lora_new_A_k = nn.Linear(dim, rank, bias=False)
        self.lora_new_B_k = nn.Linear(rank, dim, bias=False)
        self.lora_new_A_v = nn.Linear(dim, rank, bias=False)
        self.lora_new_B_v = nn.Linear(rank, dim, bias=False)

        setattr(self.lora_A_k.weight, "_is_a", True)
        setattr(self.lora_B_k.weight, "_is_b", True)
        setattr(self.lora_A_v.weight, "_is_a", True)
        setattr(self.lora_B_v.weight, "_is_b", True)
        setattr(self.lora_new_A_k.weight, "_is_new_a", True)
        setattr(self.lora_new_B_k.weight, "_is_new_b", True)
        setattr(self.lora_new_A_v.weight, "_is_new_a", True)
        setattr(self.lora_new_B_v.weight, "_is_new_b", True)

        self.delta_w_k_new_grad = None
        self.delta_w_v_new_grad = None
        self.init_param()

    def init_param(self):
        nn.init.zeros_(self.lora_A_k.weight)
        nn.init.zeros_(self.lora_B_k.weight)
        nn.init.zeros_(self.lora_A_v.weight)
        nn.init.zeros_(self.lora_B_v.weight)

        nn.init.kaiming_uniform_(self.lora_new_A_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_new_B_k.weight)
        nn.init.kaiming_uniform_(self.lora_new_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_new_B_v.weight)

    def reset_new_lora(self):
        nn.init.kaiming_uniform_(self.lora_new_A_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_new_B_k.weight)
        nn.init.kaiming_uniform_(self.lora_new_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_new_B_v.weight)

    def accumulate_and_reset_lora(self):
        self.lora_A_k.weight.data += self.lora_new_A_k.weight.data
        self.lora_B_k.weight.data += self.lora_new_B_k.weight.data
        self.lora_A_v.weight.data += self.lora_new_A_v.weight.data
        self.lora_B_v.weight.data += self.lora_new_B_v.weight.data
        self.reset_new_lora()

    def _save_grad(self, name):
        def hook(grad):
            setattr(self, name, grad)
        return hook

    def forward(self, x, use_new=True, register_hook=False):
        bsz, tokens, channels = x.shape
        qkv = self.qkv(x)

        qkv[:, :, self.dim: 2 * self.dim] += self.lora_B_k(self.lora_A_k(x))
        qkv[:, :, 2 * self.dim:] += self.lora_B_v(self.lora_A_v(x))

        if use_new:
            delta_w_k_new = self.lora_new_B_k.weight @ self.lora_new_A_k.weight
            delta_w_v_new = self.lora_new_B_v.weight @ self.lora_new_A_v.weight

            qkv[:, :, self.dim: 2 * self.dim] += x @ delta_w_k_new.t()
            qkv[:, :, 2 * self.dim:] += x @ delta_w_v_new.t()

            if register_hook:
                self.delta_w_k_new_grad = None
                self.delta_w_v_new_grad = None
                delta_w_k_new.register_hook(self._save_grad("delta_w_k_new_grad"))
                delta_w_v_new.register_hook(self._save_grad("delta_w_v_new_grad"))

        qkv = qkv.reshape(bsz, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(bsz, tokens, channels)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class BlockEWCLoRA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rank=10,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionEWCLoRA(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            rank=rank,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, use_new=True, register_hook=False):
        x = x + self.drop_path1(self.attn(self.norm1(x), use_new=use_new, register_hook=register_hook))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformerEWCLoRA(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        rank=10,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                BlockEWCLoRA(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    rank=rank,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.out_dim = embed_dim

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x, use_new=True, register_hook=False):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x, use_new=use_new, register_hook=register_hook)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, use_new=True, register_hook=False):
        return self.forward_features(x, use_new=use_new, register_hook=register_hook)

    def accumulate_and_reset_lora(self):
        for block in self.blocks:
            block.attn.accumulate_and_reset_lora()


class ViTEWCLoRA(nn.Module):
    def __init__(
        self,
        pretrained=False,
        model_name="vit_base_patch16_224_in21k",
        image_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        rank=10,
        norm_layer_eps=1e-6,
        device=None,
        **kwargs,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        self.feat = VisionTransformerEWCLoRA(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            rank=rank,
        )
        self.out_dim = self.feat.out_dim

        if pretrained:
            print(f"Using pretrained model : {model_name}")
            state_dict = timm.create_model(model_name, pretrained=True).state_dict()
            self.feat.load_state_dict(state_dict, strict=False)

    def forward(self, image, use_new=True, register_hook=False):
        return self.feat(image, use_new=use_new, register_hook=register_hook)

    def accumulate_and_reset_lora(self):
        self.feat.accumulate_and_reset_lora()


def vit_ewclora(pretrained=False, **kwargs):
    return ViTEWCLoRA(pretrained=pretrained, **kwargs)
