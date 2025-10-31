from typing import Optional
import torch
import torch.nn as nn

from sensortokenizer import SensorTokenizer


def causal_mask(sz: int, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)


class PretrainTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        seq_length: int = 120,
        patch_size: int = 20,
        stride: int = 10,
        embed_dim: int = 64,
        num_layers: int = 1,
        nhead: int = 1,
        window_fn: Optional[callable] = None,
        rot_aug: bool = True,
        rot_max_angle: Optional[float] = None,
        rot_prob: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.tokenizer = SensorTokenizer(
            in_channels=in_channels,
            patch_size=patch_size,
            stride=stride,
            window_fn=window_fn,
            rot_aug=rot_aug,
            rot_max_angle=rot_max_angle,
            rot_prob=rot_prob,
        )
        self.freq_len = patch_size // 2 + 1
        self.input_dim = in_channels * self.freq_len
        self.proj = nn.Linear(self.input_dim, embed_dim)
        self.max_patches = 0 if seq_length < patch_size else 1 + (seq_length - patch_size) // max(1, stride)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_patches, embed_dim)) if self.max_patches > 0 else None
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 2, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, in_channels * patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _raw_patches(self, x: torch.Tensor) -> torch.Tensor:
        x_unf = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        B, C, P, Lp = x_unf.shape
        return x_unf.permute(0, 2, 1, 3).contiguous().view(B, P, C * Lp)

    def _init_weights(self) -> None:
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, return_targets: bool = False):
        tokens = self.tokenizer(x)
        tokens = self.proj(tokens)
        T = tokens.shape[1]
        if self.max_patches > 0 and T > self.max_patches:
            tokens = tokens[:, : self.max_patches, :]
            T = tokens.shape[1]
        if self.pos_emb is not None:
            tokens = tokens + self.pos_emb[:, :T, :]
        attn_mask = causal_mask(T, device=tokens.device).to(torch.float32) * -1e9
        out = self.transformer(tokens, mask=attn_mask)
        out = self.norm(out)
        preds = self.output_proj(out)
        if return_targets:
            target_raw = self._raw_patches(x)[:, :T, :]
            return preds[:, :-1, :], target_raw[:, 1:, :]
        return preds