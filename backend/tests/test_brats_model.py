"""Phase 6 — architecture reconstruction.

Proves the production UNet3D is the network the checkpoint was trained with:
the same parameter count, the same channel schedule, the same state_dict keys,
and the same output geometry the notebook asserted in Section 12.4.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.inference.brats.config import TRAINING_CONFIG  # noqa: E402
from app.inference.brats.model import (  # noqa: E402
    ConvBlock3D,
    Down3D,
    UNet3D,
    Up3D,
    build_model,
    count_parameters,
)
from tests.phase6_helpers import small_config  # noqa: E402

# Printed by the notebook's own Section 12.3 cell for the trained
# configuration (base_channels=32, max_channels=320, num_stages=5).
NOTEBOOK_PARAMETER_COUNT = 18_774_756
NOTEBOOK_CHANNEL_SCHEDULE = [32, 64, 128, 256, 320]


def test_trained_architecture_matches_notebook_parameter_count() -> None:
    model = build_model(TRAINING_CONFIG.model)
    assert count_parameters(model) == NOTEBOOK_PARAMETER_COUNT


def test_trained_architecture_channel_schedule() -> None:
    model = build_model(TRAINING_CONFIG.model)
    assert model.channels == NOTEBOOK_CHANNEL_SCHEDULE
    assert list(TRAINING_CONFIG.model.channel_schedule) == NOTEBOOK_CHANNEL_SCHEDULE


def test_encoder_decoder_stage_counts() -> None:
    model = build_model(TRAINING_CONFIG.model)
    assert len(model.encoder_blocks) == 5
    assert len(model.downs) == 4
    assert len(model.decoder_blocks) == 4


def test_block_composition_matches_notebook() -> None:
    """Two 3x3x3 convs, InstanceNorm3d(affine=True), LeakyReLU(0.01), bias=False."""
    model = build_model(TRAINING_CONFIG.model)
    block = model.encoder_blocks[0]
    assert isinstance(block, ConvBlock3D)
    layers = list(block.block)
    assert isinstance(layers[0], torch.nn.Conv3d)
    assert layers[0].kernel_size == (3, 3, 3)
    assert layers[0].padding == (1, 1, 1)
    assert layers[0].bias is None
    assert isinstance(layers[1], torch.nn.InstanceNorm3d)
    assert layers[1].affine is True
    assert isinstance(layers[2], torch.nn.LeakyReLU)
    assert layers[2].negative_slope == pytest.approx(0.01)
    assert isinstance(layers[3], torch.nn.Conv3d)
    assert layers[3].bias is None


def test_downsampling_is_strided_conv_not_pooling() -> None:
    model = build_model(TRAINING_CONFIG.model)
    down = model.downs[0]
    assert isinstance(down, Down3D)
    assert isinstance(down.down, torch.nn.Conv3d)
    assert down.down.stride == (2, 2, 2)
    assert down.down.kernel_size == (2, 2, 2)
    assert not any(
        isinstance(m, (torch.nn.MaxPool3d, torch.nn.AvgPool3d)) for m in model.modules()
    )


def test_upsampling_is_transposed_conv() -> None:
    model = build_model(TRAINING_CONFIG.model)
    up = model.decoder_blocks[0]
    assert isinstance(up, Up3D)
    assert isinstance(up.up, torch.nn.ConvTranspose3d)
    assert up.up.stride == (2, 2, 2)


def test_final_conv_is_1x1x1_to_four_classes() -> None:
    model = build_model(TRAINING_CONFIG.model)
    assert isinstance(model.final_conv, torch.nn.Conv3d)
    assert model.final_conv.kernel_size == (1, 1, 1)
    assert model.final_conv.out_channels == 4


def test_no_output_activation_logits_are_returned() -> None:
    """The network emits raw logits; softmax lives in sliding_window_infer."""
    config = small_config()
    model = build_model(config.model).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 4, 32, 32, 32))
    assert output.min() < 0.0, "logits should be able to go negative"
    assert not torch.allclose(output.sum(dim=1), torch.ones_like(output.sum(dim=1)))


def test_patch_smaller_than_the_architecture_minimum_is_rejected_loudly() -> None:
    """A 5-stage network cannot accept a patch below 32 per axis.

    The bottleneck would collapse to one spatial element and InstanceNorm3d
    raises. This is asserted so the constraint is documented by a test rather
    than discovered at inference time.
    """
    config = small_config()
    model = build_model(config.model).eval()
    with torch.inference_mode(), pytest.raises(ValueError, match="spatial element"):
        model(torch.randn(1, 4, 16, 16, 16))


def test_forward_output_shape_matches_input_geometry() -> None:
    config = small_config()
    model = build_model(config.model).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 4, 32, 32, 32))
    assert tuple(output.shape) == (1, 4, 32, 32, 32)


def test_state_dict_keys_are_stable_across_construction() -> None:
    """Two independently constructed models must agree key-for-key and
    shape-for-shape, which is what makes strict checkpoint loading meaningful."""
    a = build_model(TRAINING_CONFIG.model)
    b = UNet3D(TRAINING_CONFIG.model)
    assert list(a.state_dict().keys()) == list(b.state_dict().keys())
    for key, tensor in a.state_dict().items():
        assert tensor.shape == b.state_dict()[key].shape
