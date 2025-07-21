from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict, ModelTiling, SizeRequirements
from ...util import get_seq_len, KeyCondition
from .__arch.aethernet_arch import AetherNet
from typing import Tuple

class AetherNetArch(Architecture[AetherNet]):
    def __init__(self) -> None:
        super().__init__(
            id="AetherNet",
            name="AetherNet",
            detect=KeyCondition.has_all(
                "conv_first.weight",
                "conv_after_body.weight",
                "conv_before_upsample.weight",
                "conv_last.weight",
                KeyCondition.has_any(
                    "stages.0.0.conv.lk_conv.weight",      # unfused
                    "stages.0.0.conv.fused_conv.weight",   # fused
                ),
            ),
        )

    def load(self, state_dict: StateDict) -> ImageModelDescriptor[AetherNet]:
        # Defaults from the AetherNet constructor
        in_chans: int = 3
        embed_dim: int = 96
        depths: Tuple[int, ...] = (4, 4, 4, 4)
        mlp_ratio: float = 1.5
        drop: float = 0.0
        drop_path_rate: float = 0.1
        lk_kernel: int = 13
        sk_kernel: int = 5
        scale: int = 4
        img_range: float = 1.0
        fused_init: bool = False
        quantize_residual: bool = True
        use_channel_attn: bool = True
        use_spatial_attn: bool = False
        norm_type: str = "deployment"
        res_scale: float = 0.1

        # Detections
        in_chans = state_dict["conv_first.weight"].shape[1]
        embed_dim = state_dict["conv_first.weight"].shape[0]

        num_stages = get_seq_len(state_dict, "stages")
        depths = tuple(get_seq_len(state_dict, f"stages.{i}") for i in range(num_stages))

        # Check for fused or unfused model
        if "stages.0.0.conv.fused_conv.weight" in state_dict:
            fused_init = True
            lk_kernel = state_dict["stages.0.0.conv.fused_conv.weight"].shape[2]
            # sk_kernel cannot be determined from fused model, so we keep the default
        else:
            fused_init = False
            lk_kernel = state_dict["stages.0.0.conv.lk_conv.weight"].shape[2]
            sk_kernel = state_dict["stages.0.0.conv.sk_conv.weight"].shape[2]

        mlp_ratio = (
            state_dict["stages.0.0.ffn.conv_gate.weight"].shape[0] / embed_dim
        )

        if "stages.0.0.norm.running_mean" in state_dict:
            norm_type = "deployment"
        else:
            norm_type = "layernorm"

        use_channel_attn = "stages.0.0.channel_attn.fc.0.weight" in state_dict
        use_spatial_attn = "stages.0.0.spatial_attn.conv.weight" in state_dict
        quantize_residual = "stages.0.0.res_dequant.scale" in state_dict

        # Detect scale from upsampler
        num_upsample_blocks = get_seq_len(state_dict, "upsample.blocks")

        if num_upsample_blocks == 0:
            # No blocks means scale is 1x (Identity layers have no state)
            scale = 1
        elif "upsample.blocks.0.0.weight" in state_dict:
            # Check for the distinctive 3x scale first.
            # The conv layer for 3x upsampling multiplies channels by 9.
            # The number of input channels to the upsampler is embed_dim.
            if state_dict["upsample.blocks.0.0.weight"].shape[0] == embed_dim * 9:
                scale = 3
            else:
                # Otherwise, it's a power of 2. For AetherNet, each block is a 2x upscale.
                scale = 2 ** num_upsample_blocks
        else:
            # Fallback for unexpected cases, assume 1x.
            scale = 1

        # Presets and tags based on embed_dim and num_stages
        if embed_dim <= 64 and num_stages == 3:
            arch_name = "aether_tiny"
            res_scale = 0.2
        elif embed_dim <= 96 and num_stages == 4:
            arch_name = "aether_small"
        elif embed_dim <= 128 and num_stages == 4:
            arch_name = "aether_medium"
        elif embed_dim <= 180 and num_stages == 5:
            arch_name = "aether_large"
        else:
            arch_name = "custom"

        tags = [arch_name]
        if fused_init: tags.append("fused")
        # Check if model comes from QAT based on a key from quantization stubs
        if "quant.scale" in state_dict: tags.append("qat")

        model = AetherNet(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path_rate=drop_path_rate,
            lk_kernel=lk_kernel,
            sk_kernel=sk_kernel,
            scale=scale,
            img_range=img_range,
            fused_init=fused_init,
            quantize_residual=quantize_residual,
            use_channel_attn=use_channel_attn,
            use_spatial_attn=use_spatial_attn,
            norm_type=norm_type,
            res_scale=res_scale,
        )

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
            tiling=ModelTiling.SUPPORTED,
            size_requirements=SizeRequirements(multiple_of=8),
        )
