"""DeiT-Tiny adapted to single-channel astrophysical lensing maps.

Adaptations
-----------
1. Patch embedding: the pretrained (192, 3, 16, 16) projection is collapsed to
   (192, 1, 16, 16) by summing over RGB, so ImageNet features are preserved.
2. Resolution: 150 is not a multiple of the 16-px patch, so images are zero-padded
   (in intensity space, where the background is ~0) to 160 = 10 x 10 patches. The
   pretrained 14 x 14 position embeddings are bicubically resampled to 10 x 10 by timm.
   This uses 100 tokens instead of the 196 needed by upscaling to 224, which is ~2x
   cheaper and avoids interpolating the substructure signal.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import IMG_SIZE, NUM_CLASSES
from .common import Standardize, rgb_to_single_channel

PATCH = 16
PADDED = ((IMG_SIZE + PATCH - 1) // PATCH) * PATCH   # 160


class DeiTTinyClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, drop_path: float = 0.1):
        super().__init__()
        self.norm = Standardize()
        self.backbone = timm.create_model(
            "deit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes,
            img_size=PADDED, drop_path_rate=drop_path,
        )
        proj = self.backbone.patch_embed.proj
        new = nn.Conv2d(1, proj.out_channels, proj.kernel_size, proj.stride, bias=proj.bias is not None)
        with torch.no_grad():
            new.weight.copy_(rgb_to_single_channel(proj.weight))
            if proj.bias is not None:
                new.bias.copy_(proj.bias)
        self.backbone.patch_embed.proj = new
        self.head = self.backbone.head          # exposed so the trainer can give it its own LR

    def pad(self, x):
        p = PADDED - x.shape[-1]
        return F.pad(x, (p // 2, p - p // 2, p // 2, p - p // 2))

    def forward(self, x):
        return self.backbone(self.norm(self.pad(x)))

    @torch.no_grad()
    def attention_rollout(self, x, discard_ratio: float = 0.0):
        """Attention rollout (Abnar & Zuidema, 2020) from the CLS token to each patch.

        Returns (B, 10, 10) maps; upsample to 150x150 for overlays.
        """
        maps = []

        def hook(module, inp, _out):
            t = inp[0]
            b, n, c = t.shape
            qkv = module.qkv(t).reshape(b, n, 3, module.num_heads, c // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            q, k = module.q_norm(q), module.k_norm(k)
            maps.append(((q @ k.transpose(-2, -1)) * module.scale).softmax(-1).mean(1))

        handles = [blk.attn.register_forward_hook(hook) for blk in self.backbone.blocks]
        try:
            self.eval()(x)
        finally:
            for h in handles:
                h.remove()

        n = maps[0].shape[-1]
        rollout = torch.eye(n, device=x.device).expand_as(maps[0]).clone()
        for a in maps:
            if discard_ratio:
                flat = a.flatten(1)
                k = int(flat.shape[1] * discard_ratio)
                idx = flat.topk(k, dim=1, largest=False).indices
                flat.scatter_(1, idx, 0)
            a = a + torch.eye(n, device=x.device)
            a = a / a.sum(-1, keepdim=True)
            rollout = a @ rollout
        cls = rollout[:, 0, 1:]                                  # CLS -> patches
        side = int(cls.shape[-1] ** 0.5)
        cls = cls.reshape(-1, side, side)
        return cls / cls.amax(dim=(-2, -1), keepdim=True)
