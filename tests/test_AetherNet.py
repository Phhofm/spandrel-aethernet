from spandrel.architectures.AetherNet import AetherNetArch, AetherNet  # Corrected import path
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
    """
    Tests that the load function can correctly detect parameters for various
    model configurations.
    """
    assert_loads_correctly(
        AetherNetArch(),
        # Test a variety of scales
        lambda: AetherNet(scale=1),
        lambda: AetherNet(scale=2),
        lambda: AetherNet(scale=3),
        lambda: AetherNet(scale=4),
        # This now explicitly tests the "tiny" preset's conditions
        lambda: AetherNet(embed_dim=64, depths=(3, 3, 3), res_scale=0.2),
        # Test different norm_type
        lambda: AetherNet(norm_type="layernorm"),
        # Test fused vs. unfused.
        lambda: AetherNet(fused_init=True),
        lambda: AetherNet(fused_init=False),
        # Test other boolean flags
        lambda: AetherNet(use_channel_attn=False),
        lambda: AetherNet(use_spatial_attn=True),
        lambda: AetherNet(quantize_residual=False),
        # Test a complex combination
        lambda: AetherNet(
            scale=2,
            embed_dim=128,
            depths=(5, 5, 5, 5),
            norm_type="layernorm",
            use_spatial_attn=True,
        ),
        # Add this argument to ignore the undetectable parameter
        ignore_parameters={"quantize_residual"},
    )

def test_size_requirements():
    """
    Tests the size requirements of a default AetherNet model and runs an
    inference test.
    """
    # Create the model descriptor in memory
    arch = AetherNetArch()
    model = AetherNet(scale=4)
    state_dict = model.state_dict()
    loaded_model_descriptor = arch.load(state_dict)

    # Check size requirements
    assert loaded_model_descriptor.size_requirements.multiple_of == 4

    # Create a dummy ModelFile just for naming the output of the inference test
    dummy_file = ModelFile(name="AetherNet_dummy_4x_size_req")

    assert_image_inference(
        dummy_file,
        loaded_model_descriptor,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )

def test_AetherNet_tiny_x4(snapshot):
    arch = AetherNetArch()
    model_fn = lambda: AetherNet(embed_dim=64, depths=(3, 3, 3), scale=4, res_scale=0.2)

    # Manually create the model descriptor
    model_instance = model_fn()
    state_dict = model_instance.state_dict()
    loaded_model_descriptor = arch.load(state_dict)

    assert loaded_model_descriptor == snapshot(exclude=disallowed_props)
    assert isinstance(loaded_model_descriptor.model, AetherNet)

    # Create a dummy ModelFile for the inference test output name
    dummy_file = ModelFile(name="AetherNet_tiny_x4")
    assert_image_inference(
        dummy_file,
        loaded_model_descriptor,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )

def test_AetherNet_small_x4(snapshot):
    arch = AetherNetArch()
    model_fn = lambda: AetherNet(embed_dim=96, depths=(4, 4, 4, 4), scale=4)

    model_instance = model_fn()
    state_dict = model_instance.state_dict()
    loaded_model_descriptor = arch.load(state_dict)

    assert loaded_model_descriptor == snapshot(exclude=disallowed_props)
    assert isinstance(loaded_model_descriptor.model, AetherNet)

    dummy_file = ModelFile(name="AetherNet_small_x4")
    assert_image_inference(
        dummy_file,
        loaded_model_descriptor,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )

def test_AetherNet_medium_x4(snapshot):
    arch = AetherNetArch()
    model_fn = lambda: AetherNet(embed_dim=128, depths=(6, 6, 6, 6), scale=4)

    model_instance = model_fn()
    state_dict = model_instance.state_dict()
    loaded_model_descriptor = arch.load(state_dict)

    assert loaded_model_descriptor == snapshot(exclude=disallowed_props)
    assert isinstance(loaded_model_descriptor.model, AetherNet)

    dummy_file = ModelFile(name="AetherNet_medium_x4")
    assert_image_inference(
        dummy_file,
        loaded_model_descriptor,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )

def test_AetherNet_large_x4(snapshot):
    arch = AetherNetArch()
    model_fn = lambda: AetherNet(embed_dim=180, depths=(8, 8, 8, 8, 8), scale=4, use_spatial_attn=True)

    model_instance = model_fn()
    state_dict = model_instance.state_dict()
    loaded_model_descriptor = arch.load(state_dict)

    assert loaded_model_descriptor == snapshot(exclude=disallowed_props)
    assert isinstance(loaded_model_descriptor.model, AetherNet)

    dummy_file = ModelFile(name="AetherNet_large_x4")
    assert_image_inference(
        dummy_file,
        loaded_model_descriptor,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )

def test_real_trained_model_loading():
    """
    This is the most important test. It uses a real trained model file
    to verify that the detection and loading process works end-to-end.
    """
    # This tells the test to look for the model in `tests/models`
    #file = ModelFile(name="net_g_185000.pth")
    file = ModelFile(name="net_g_185000_aether_large_fp32.pth")


    # This uses the full ModelLoader, which first DETECTS the architecture
    # from the MAIN_REGISTRY, and then calls the `load` function.
    # This is the exact same process chaiNNer uses.
    model = file.load_model()

    # If the test reaches here, it means detection and loading were successful.
    # We can add an assertion just to be sure.
    assert isinstance(model.model, AetherNet)
    print("Successfully loaded the real model!")
