import torch
import torch.nn as nn

from .pretrain import PretrainTransformer

class FineTuneModel(nn.Module):
    def __init__(self, pretrain_model: PretrainTransformer, n_class: int, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = pretrain_model.tokenizer
        self.proj = pretrain_model.proj
        self.pos_emb = getattr(pretrain_model, "pos_emb", None)
        self.max_patches = pretrain_model.max_patches
        self.transformer = pretrain_model.transformer
        self.norm = pretrain_model.norm
        embed_dim = pretrain_model.embed_dim
        nhead = pretrain_model.nhead
        self.post_encoder_dropout = nn.Dropout(dropout) if (dropout and dropout > 0.0) else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
            batch_first=True,
        )
        self.classifier_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier_mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_class),
        )

    def _raw_length_clip(self, T: int) -> int:
        if self.max_patches is not None and self.max_patches > 0:
            return min(T, self.max_patches)
        return T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        tokens = self.proj(tokens)
        T = self._raw_length_clip(tokens.shape[1])
        tokens = tokens[:, :T, :]
        if self.pos_emb is not None:
            tokens = tokens + self.pos_emb[:, :T, :]
        out = self.transformer(tokens, mask=None)
        out = self.norm(out)
        out = self.post_encoder_dropout(out)
        out = self.classifier_transformer(out, mask=None)
        pooled = out.mean(dim=1)
        logits = self.classifier_mlp_head(pooled)
        return logits