# -*- coding: utf-8 -*-
"""
https://github.com/Phhofm/aethernet

AetherNet: Ultra-Fast Super-Resolution Network with QAT Support

Core architecture definition featuring:
- Structural reparameterization for inference efficiency
- Quantization-aware training (INT8, FP16) support
- Deployment-friendly normalization
- Multi-scale feature fusion
- Adaptive upsampling

Designed for easy integration into super-resolution frameworks like Spandrel,
neosr, and traiNNer-redux. Minimal dependencies - only requires PyTorch.
"""

import math
import warnings
from copy import deepcopy
from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
import torch.ao.quantization as tq


class DropPath(nn.Module):
    """
    Stochastic Depth implementation compatible with ONNX export.

    During training, randomly drops entire sample paths with given probability.
    During inference, acts as identity function.

    Args:
        drop_prob: Probability of dropping a path (0.0 = no drop)
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError("drop_prob must be between 0.0 and 1.0")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize to 0 or 1
        return x.div(keep_prob) * random_tensor


class ReparamLargeKernelConv(nn.Module):
    """
    Efficient large kernel convolution using structural reparameterization.

    Combines large and small kernel convolutions during training, fuses them
    into a single convolution for inference efficiency.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Main kernel size (must be odd)
        stride: Convolution stride
        groups: Number of groups
        small_kernel: Parallel small kernel size (must be odd)
        fused_init: Initialize in fused mode
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        if kernel_size % 2 == 0 or small_kernel % 2 == 0:
            raise ValueError("Kernel sizes must be odd for symmetrical padding")
        if groups <= 0:
            raise ValueError("Number of groups must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2
        self.small_kernel = small_kernel
        self.fused = fused_init
        self.is_quantized = False

        if self.fused:
            # Directly initialize fused convolution
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                padding=self.padding, groups=groups, bias=True
            )
        else:
            # Training branch: large kernel convolution without bias
            self.lk_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                self.padding, groups=groups, bias=False
            )
            # Training branch: small kernel convolution without bias
            self.sk_conv = nn.Conv2d(
                in_channels, out_channels, small_kernel, stride,
                small_kernel // 2, groups=groups, bias=False
            )
            # Bias parameters for each convolution
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return self.fused_conv(x)

        # Training forward: apply both convolutions and add their outputs
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return (lk_out + self.lk_bias.view(1, -1, 1, 1) +
                sk_out + self.sk_bias.view(1, -1, 1, 1))

    def fuse(self):
        """Fuse the two convolutions and their biases into a single convolution for inference."""
        if self.fused:
            return

        # Pad the small kernel to the size of the large kernel
        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = F.pad(self.sk_conv.weight, [pad] * 4)
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias

        # Create the fused convolution
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, padding=self.padding, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias

        # Preserve quantization configuration if present
        if self.is_quantized and hasattr(self.lk_conv, 'qconfig'):
            self.fused_conv.qconfig = self.lk_conv.qconfig

        # Cleanup the training-specific parameters
        del self.lk_conv, self.sk_conv, self.lk_bias, self.sk_bias
        self.fused = True


class GatedConvFFN(nn.Module):
    """
    Gated Feed-Forward Network for enhanced feature transformation.

    Uses SiLU activation with temperature scaling and quantization support.

    Args:
        in_channels: Input channels
        mlp_ratio: Hidden dimension multiplier (default: 1.5)
        drop: Dropout probability (default: 0.0)
    """
    def __init__(self, in_channels: int, mlp_ratio: float = 1.5, drop: float = 0.):
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= drop <= 1.0:
            raise ValueError("drop probability must be between 0 and 1")

        hidden_channels = int(in_channels * mlp_ratio)

        self.conv_gate = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv_main = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 1)
        self.drop2 = nn.Dropout(drop)
        self.temperature = nn.Parameter(torch.ones(1))
        self.is_quantized = False
        self.quant_mul = torch.nn.quantized.FloatFunctional()

        # Quantization stubs for activation and temperature scaling
        self.act_dequant = tq.DeQuantStub()
        self.act_quant = tq.QuantStub()
        self.temp_dequant = tq.DeQuantStub()
        self.temp_quant = tq.QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate branch with temperature scaling
        gate_unscaled = self.conv_gate(x)
        if self.is_quantized:
            # In quantized mode, do temperature scaling in float and requantize
            gate_float = self.temp_dequant(gate_unscaled)
            gate_scaled = gate_float * self.temperature
            gate = self.temp_quant(gate_scaled)
        else:
            gate = gate_unscaled * self.temperature

        # Main branch
        main = self.conv_main(x)

        # Apply activation in float domain for quantization compatibility
        gate = self.act_dequant(gate)
        activated = self.act(gate)
        activated = self.act_quant(activated)

        # Element-wise multiplication
        if x.dtype == torch.float16:
            # Handle FP16 with FP32 precision to avoid overflow
            x = activated.float() * main.float()
            x = x.half()
        elif self.is_quantized:
            # Use quantized multiplication
            x = self.quant_mul.mul(activated, main)
        else:
            x = activated * main

        x = self.drop1(x)
        x = self.conv_out(x)
        return x


class DynamicChannelScaling(nn.Module):
    """
    Efficient Channel Attention (Squeeze-and-Excitation) mechanism.

    Args:
        dim: Input dimension (number of channels)
        reduction: Channel reduction ratio (default: 8)
    """
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        if dim <= reduction:
            raise ValueError(f"Reduction ratio {reduction} is too large for {dim} channels")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.avg_pool(x))
        if self.is_quantized:
            return self.quant_mul.mul(x, scale)
        return x * scale


class SpatialAttention(nn.Module):
    """
    Lightweight spatial attention module.

    Args:
        kernel_size: Convolution kernel size (default: 7)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd for symmetrical padding")

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aggregate spatial information using max and average pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = self.sigmoid(self.conv(concat))
        if self.is_quantized:
            return self.quant_mul.mul(x, attention_map)
        return x * attention_map


class DeploymentNorm(nn.Module):
    """
    Deployment-friendly normalization layer with fusion support.

    Maintains running statistics during training, converts to
    simple affine transform for inference.

    Args:
        channels: Number of channels
        eps: Numerical stability epsilon (default: 1e-4)
    """
    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        if channels <= 0:
            raise ValueError("Number of channels must be positive")
        if eps <= 0:
            raise ValueError("Epsilon must be positive")

        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
        self.fused = False
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return x * self.weight + self.bias

        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            with torch.no_grad():
                # Update running statistics with momentum
                self.running_mean.mul_(0.9).add_(mean, alpha=0.1)
                self.running_var.mul_(0.9).add_(var, alpha=0.1)
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

    def fuse(self):
        """Fuse the normalization into a single affine transform for inference."""
        if self.fused:
            return

        # Compute fused scale and shift
