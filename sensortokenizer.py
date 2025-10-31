from typing import Optional, Tuple
import torch
import torch.nn as nn


class SensorTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        stride: int,
        window_fn: Optional[callable] = None,
        rot_aug: bool = True,
        rot_max_angle: Optional[float] = None,
        rot_prob: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.stride = stride
        self.window_fn = window_fn
        self.rot_aug = rot_aug
        self.rot_max_angle = rot_max_angle
        self.rot_prob = rot_prob
        self.fft_len = patch_size // 2 + 1

    @staticmethod
    def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        x, y, z, w = q.unbind(-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        m00 = 1 - 2 * (yy + zz)
        m01 = 2 * (xy - wz)
        m02 = 2 * (xz + wy)
        m10 = 2 * (xy + wz)
        m11 = 1 - 2 * (xx + zz)
        m12 = 2 * (yz - wx)
        m20 = 2 * (xz - wy)
        m21 = 2 * (yz + wx)
        m22 = 1 - 2 * (xx + yy)
        return torch.stack(
            [
                torch.stack([m00, m01, m02], dim=-1),
                torch.stack([m10, m11, m12], dim=-1),
                torch.stack([m20, m21, m22], dim=-1),
            ],
            dim=-2,
        )

    @staticmethod
    def _rand_rotmats_uniform(bp: Tuple[int, int], device) -> torch.Tensor:
        B, P = bp
        u1 = torch.rand(B, P, device=device)
        u2 = torch.rand(B, P, device=device)
        u3 = torch.rand(B, P, device=device)
        r1 = torch.sqrt(1.0 - u1)
        r2 = torch.sqrt(u1)
        th1 = 2 * torch.pi * u2
        th2 = 2 * torch.pi * u3
        qx = r1 * torch.sin(th1)
        qy = r1 * torch.cos(th1)
        qz = r2 * torch.sin(th2)
        qw = r2 * torch.cos(th2)
        q = torch.stack([qx, qy, qz, qw], dim=-1)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        return SensorTokenizer._quat_to_rotmat(q)

    @staticmethod
    def _rand_rotmats_bounded(bp: Tuple[int, int], max_angle: float, device) -> torch.Tensor:
        B, P = bp
        v = torch.randn(B, P, 3, device=device)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        theta = (torch.rand(B, P, device=device) * 2 - 1) * max_angle
        ct = torch.cos(theta)[..., None]
        st = torch.sin(theta)[..., None]
        vx, vy, vz = v.unbind(-1)
        K = torch.stack(
            [
                torch.zeros_like(vx),
                -vz,
                vy,
                vz,
                torch.zeros_like(vx),
                -vx,
                -vy,
                vx,
                torch.zeros_like(vx),
            ],
            dim=-1,
        ).reshape(B, P, 3, 3)
        I = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, P, 3, 3)
        v_outer = v[..., :, None] * v[..., None, :]
        return I * ct + K * st + (1.0 - ct) * v_outer

    def _maybe_apply_rotation(self, x_unf: torch.Tensor) -> torch.Tensor:
        if (not self.rot_aug) or self.in_channels < 6:
            return x_unf
        B, _, P, _ = x_unf.shape
        device = x_unf.device
        R = (
            self._rand_rotmats_uniform((B, P), device)
            if self.rot_max_angle is None
            else self._rand_rotmats_bounded((B, P), float(self.rot_max_angle), device)
        )
        if self.rot_prob < 1.0:
            mask = (torch.rand(B, P, device=device) < self.rot_prob).to(x_unf.dtype)
            I = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, P, 3, 3)
            R = R * mask[..., None, None] + I * (1.0 - mask[..., None, None])

        acc = x_unf[:, 0:3, :, :].permute(0, 2, 1, 3)
        acc = torch.einsum("bpij,bpjl->bpil", R, acc).permute(0, 2, 1, 3)
        gyro = x_unf[:, 3:6, :, :].permute(0, 2, 1, 3)
        gyro = torch.einsum("bpij,bpjl->bpil", R, gyro).permute(0, 2, 1, 3)
        out = x_unf.clone()
        out[:, 0:3, :, :] = acc
        out[:, 3:6, :, :] = gyro
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        if L < self.patch_size:
            raise ValueError(f"patch_size {self.patch_size} is larger than sequence length {L}")
        x_unf = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        if self.window_fn is not None:
            w = self.window_fn(self.patch_size).to(x_unf.device)
            x_unf = x_unf * w.view(1, 1, 1, -1)
        x_unf = self._maybe_apply_rotation(x_unf)
        x_fft = torch.fft.rfft(x_unf, n=self.patch_size, dim=-1)
        logmag = torch.log1p(torch.abs(x_fft))
        B, C, P, F = logmag.shape
        return logmag.permute(0, 2, 1, 3).contiguous().view(B, P, C * F)