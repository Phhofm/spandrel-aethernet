import math
from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict, ModelTiling, SizeRequirements
from .__arch.aethernet_arch import AetherNet
from typing import Tuple

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
        state_dict.pop("pt_version", None)
        state_dict.pop("model_version", None)

        in_chans = state_dict["conv_first.weight"].shape[1]
        embed_dim = state_dict["conv_first.weight"].shape[0]
        num_stages = get_seq_len(state_dict, "stages")
        depths = tuple(get_seq_len(state_dict, f"stages.{i}") for i in range(num_stages))

        fused_init = "stages.0.0.conv.fused_conv.weight" in state_dict
        if fused_init:
            lk_kernel = state_dict["stages.0.0.conv.fused_conv.weight"].shape[2]
            sk_kernel = 5  # Default, cannot be detected from fused model
        else:
            lk_kernel = state_dict["stages.0.0.conv.lk_conv.weight"].shape[2]
            sk_kernel = state_dict["stages.0.0.conv.sk_conv.weight"].shape[2]

        mlp_ratio = state_dict["stages.0.0.ffn.conv_gate.weight"].shape[0] / embed_dim

        norm_weight_shape = state_dict["stages.0.0.norm.weight"].shape
        if len(norm_weight_shape) == 4:
            norm_type = "deployment"
        elif len(norm_weight_shape) == 1:
            norm_type = "layernorm"
        else:
            raise ValueError(f"Unexpected norm weight shape: {norm_weight_shape}")

        use_channel_attn = "stages.0.0.channel_attn.fc.0.weight" in state_dict
        use_spatial_attn = "stages.0.0.spatial_attn.conv.weight" in state_dict

        # Correct scale detection that handles both arch versions
        scale = int(state_dict["scale_tensor"].item()) if "scale_tensor" in state_dict else 4

        res_scale = 0.1
        if embed_dim <= 64:
            arch_name = "aether_tiny"
            res_scale = 0.2
        elif embed_dim <= 96:
            arch_name = "aether_small"
        elif embed_dim <= 128:
            arch_name = "aether_medium"
        elif embed_dim <= 180:
            arch_name = "aether_large"
        else:
            arch_name = "custom"
        tags = [arch_name]

        model = AetherNet(
            in_chans=in_chans, embed_dim=embed_dim, depths=depths, mlp_ratio=mlp_ratio,
            lk_kernel=lk_kernel, sk_kernel=sk_kernel, scale=scale,
            use_channel_attn=use_channel_attn, use_spatial_attn=use_spatial_attn,
            norm_type=norm_type, res_scale=res_scale,
            fused_init=fused_init,  # Pass the detected fusion state here
        )

        if (scale & (scale - 1)) == 0 and scale > 0:
            multiple_of = scale
        else:
            multiple_of = 1

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
            output_channels=in_chans,
            size_requirements=SizeRequirements(multiple_of=multiple_of),
        )
