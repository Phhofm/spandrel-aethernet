import torch
from spandrel.architectures.AetherNet.__arch.aethernet_arch import AetherNet

# Load your release model
release_model_path = "tests/models/net_g_185000_aether_large_fp32.pth"
checkpoint = torch.load(release_model_path, map_location="cpu")
state_dict = checkpoint['state_dict']

# Check for the fused key
fused_key = "stages.0.0.conv.fused_conv.weight"
print(f"'{fused_key}' in state_dict: {fused_key in state_dict}")

# Check for unfused keys
unfused_key = "stages.0.0.conv.lk_conv.weight"
print(f"'{unfused_key}' in state_dict: {unfused_key in state_dict}")
