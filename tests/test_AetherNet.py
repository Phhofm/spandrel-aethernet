import torch
from spandrel.architectures.AetherNet import AetherNet, AetherNetArch
from spandrel.util import get_seq_len

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    # Test unfused
    assert_loads_correctly(
        AetherNetArch(),
        lambda: AetherNet(scale=2, fused_init=False),
        lambda: AetherNet(scale=4, embed_dim=64, fused_init=False),
        lambda: AetherNet(scale=3, depths=(2, 2, 2), fused_init=False),
        lambda: AetherNet(scale=2, norm_type="layernorm", fused_init=False),
    )
    # Test fused
    assert_loads_correctly(
        AetherNetArch(),
        lambda: AetherNet(scale=2, fused_init=True),
        lambda: AetherNet(scale=4, embed_dim=64, fused_init=True),
    )

def test_size_requirements():
    file = ModelFile.from_options(AetherNetArch(), lambda: AetherNet(scale=4))
    model = file.load_model()
    assert_image_inference(
        file,
        model,
        [TestImage.SR_8, TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )

def test_AetherNet_tiny_x4(snapshot):
    model_fn = lambda: AetherNet(embed_dim=64, depths=(3, 3, 3), scale=4, res_scale=0.2)
    file = ModelFile.from_options(AetherNetArch(), model_fn)
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, AetherNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16],
    )

def test_AetherNet_small_x4(snapshot):
    model_fn = lambda: AetherNet(embed_dim=96, depths=(4, 4, 4, 4), scale=4)
    file = ModelFile.from_options(AetherNetArch(), model_fn)
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, AetherNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16],
    )

def test_AetherNet_medium_x4(snapshot):
    model_fn = lambda: AetherNet(embed_dim=128, depths=(6, 6, 6, 6), scale=4)
    file = ModelFile.from_options(AetherNetArch(), model_fn)
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, AetherNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16],
    )

def test_AetherNet_large_x4(snapshot):
    model_fn = lambda: AetherNet(embed_dim=180, depths=(8, 8, 8, 8, 8), scale=4, use_spatial_attn=True)
    file = ModelFile.from_options(AetherNetArch(), model_fn)
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, AetherNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16],
    )
