from __future__ import annotations

import torch
from torch import nn

TRANS_NAS_BENCH_101 = ["none", "nor_conv_1x1", "skip_connect", "nor_conv_3x3"]

# OPS defines operations for micro cell structures
OPS = {
    "none": lambda C_in, C_out, stride: Zero(C_in, C_out, stride),
    "nor_conv_1x1": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in, C_out, (1, 1), stride, (0, 0), (1, 1), affine, track_running_stats
    ),
    "skip_connect": lambda C_in, C_out, stride, affine, track_running_stats: Identity()
    if (stride == 1 and C_in == C_out)
    else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
    "nor_conv_3x3": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in, C_out, (3, 3), stride, (1, 1), (1, 1), affine, track_running_stats
    ),
}


class ReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
        dilation: int | tuple[int, int],
        affine: bool,
        track_running_stats: bool,
        activation: str = "relu",
    ):
        super().__init__()
        if activation == "leaky":
            ops = [nn.LeakyReLU(0.2, False)]
        elif activation == "relu":
            ops = [nn.ReLU(inplace=False)]
        else:
            raise ValueError(f"invalid activation {activation}")
        ops += [
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        ]
        self.ops = nn.Sequential(*ops)
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)  # type: ignore

    def extra_repr(self) -> str:
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Zero(nn.Module):
    def __init__(self, C_in: int, C_out: int, stride: int):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            return x[:, :, :: self.stride, :: self.stride].mul(0.0)

        shape = list(x.shape)
        shape[1] = self.C_out
        zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
        return zeros

    def extra_repr(self) -> str:
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class FactorizedReduce(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        stride: int | tuple[int, int],
        affine: bool,
        track_running_stats: bool,
    ):
        super().__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        assert C_out % 2 == 0, f"C_out : {C_out}"
        C_outs = [C_out // 2, C_out - C_out // 2]
        self.convs = nn.ModuleList()
        for i in range(2):
            self.convs.append(
                nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False)
            )
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.bn = nn.BatchNorm2d(
            C_out, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        # print(self.convs[0](x).shape, self.convs[1](y[:,:,1:,1:]).shape)
        # print(out.shape)

        out = self.bn(out)
        return out

    def extra_repr(self) -> str:
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)
