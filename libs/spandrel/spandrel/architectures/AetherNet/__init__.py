import math
from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict, SizeRequirements
from .arch import AetherNet  # Updated import path
from typing import Tuple, List

from spandrel.util import KeyCondition, get_seq_len

class AetherNetArch(Architecture[AetherNet]):
    def __init__(self) -> None:
        super().__init__(
            id="AetherNet",
            name="AetherNet",
            detect=KeyCondition.has_all(
                "conv_first.weight",
                "conv_last.weight",
                "stages.0.0.norm.weight",  # Common key present in all variations
                KeyCondition.has_any(
                    "stages.0.0.conv.lk_conv.weight",  # For unfused models
                    "stages.0.0.conv.fused_conv.weight",  # For fused models
                ),
            ),
        )

    def load(self, state_dict: StateDict) -> ImageModelDescriptor[AetherNet]:
        # Remove version keys if present
        state_dict.pop("pt_version", None)
        state_dict.pop("model_version", None)

        # Extract input/output channels
        in_chans = state_dict["conv_first.weight"].shape[1]
        out_chans = state_dict["conv_last.weight"].shape[0]
        embed_dim = state_dict["conv_first.weight"].shape[0]

        # Get model structure
        num_stages = get_seq_len(state_dict, "stages")
        depths = tuple(get_seq_len(state_dict, f"stages.{i}") for i in range(num_stages))

        # Detect fusion status
        fused_init = "stages.0.0.conv.fused_conv.weight" in state_dict
        if fused_init:
            lk_kernel = state_dict["stages.0.0.conv.fused_conv.weight"].shape[2]
            sk_kernel = 5
        else:
            lk_kernel = state_dict["stages.0.0.conv.lk_conv.weight"].shape[2]
            sk_kernel = state_dict["stages.0.0.conv.sk_conv.weight"].shape[2]

        mlp_ratio = state_dict["stages.0.0.ffn.conv_gate.weight"].shape[0] / embed_dim

        # Determine normalization type
        norm_weight_shape = state_dict["stages.0.0.norm.weight"].shape
        if len(norm_weight_shape) == 4:
            norm_type = "deployment"
        elif len(norm_weight_shape) == 1:
            norm_type = "layernorm"
        else:
            raise ValueError(f"Unexpected norm weight shape: {norm_weight_shape}")

        # Detect attention modules
        use_channel_attn = "stages.0.0.channel_attn.fc.0.weight" in state_dict
        use_spatial_attn = "stages.0.0.spatial_attn.conv.weight" in state_dict

        # Extract scale factor
        scale = int(state_dict["scale_tensor"].item())

        # Determine architecture variant
        tags: List[str] = []
        if "arch_name" in state_dict:
            # Use explicitly set name if available
            arch_name = state_dict["arch_name"]
            tags.append(arch_name)
        else:
            # Fallback to embedding dimension-based naming
            if embed_dim <= 64:
                tags.append("aether_tiny")
            elif embed_dim <= 96:
                tags.append("aether_small")
            elif embed_dim <= 128:
                tags.append("aether_medium")
            elif embed_dim <= 180:
                tags.append("aether_large")
            else:
                tags.append("custom")

        # Add attention info to tags
        if use_channel_attn:
            tags.append("channel_attn")
        if use_spatial_attn:
            tags.append("spatial_attn")

        # Add fusion info to tags
        tags.append("fused" if fused_init else "unfused")

        # Check for quantization
        is_qat = any("fake_quant" in k for k in state_dict)
        if is_qat:
            tags.append("qat")

        # Create model instance
        model = AetherNet(
            in_chans=in_chans, embed_dim=embed_dim, depths=depths, mlp_ratio=mlp_ratio,
            lk_kernel=lk_kernel, sk_kernel=sk_kernel, scale=scale,
            use_channel_attn=use_channel_attn, use_spatial_attn=use_spatial_attn,
            norm_type=norm_type, res_scale=0.1,
            fused_init=fused_init,
        )

        # Prepare for QAT if needed
        if is_qat:
            model.prepare_qat()

        # Set size requirements
        multiple_of = scale if (scale & (scale - 1)) == 0 and scale > 0 else 1

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="SR" if scale > 1 else "Restoration",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_chans,
            output_channels=out_chans,
            size_requirements=SizeRequirements(multiple_of=multiple_of),
        )

# Add this line to export both the architecture and model
__all__ = ["AetherNetArch", "AetherNet"]
