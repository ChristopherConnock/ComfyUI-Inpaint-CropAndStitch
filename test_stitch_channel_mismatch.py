"""
Regression test for the RGBA-inpaint channel mismatch in InpaintStitchImproved.

Reproduces: feeding a 4-channel (RGBA) inpainted image -- as returned by some
external API nodes, e.g. Gemini image nodes -- into the stitch node while the
original/canvas image is 3-channel RGB. The broadcast blend
    blended = mask * inpainted + (1 - mask) * canvas
then fails with:
    RuntimeError: The size of tensor a (4) must match the size of tensor b (3)
                  at non-singleton dimension 3

This module imports inpaint_cropandstitch.py directly. ComfyUI's own modules
(comfy.*, nodes) are not present in a bare checkout, so we stub the few names
the module references at import time before importing it.

Run with a python that has torch/torchvision/scipy/pillow/numpy:
    python test_stitch_channel_mismatch.py
"""
import sys
import types


def _install_comfy_stubs():
    """Stub the ComfyUI modules imported at module load so we can import the
    node code standalone. Only import-time symbols need to exist; the CPU code
    path under test never calls into them."""
    if 'comfy' not in sys.modules:
        comfy = types.ModuleType('comfy')
        comfy_utils = types.ModuleType('comfy.utils')
        comfy_mm = types.ModuleType('comfy.model_management')
        comfy_mm.get_torch_device = lambda: __import__('torch').device('cpu')
        comfy.utils = comfy_utils
        comfy.model_management = comfy_mm
        sys.modules['comfy'] = comfy
        sys.modules['comfy.utils'] = comfy_utils
        sys.modules['comfy.model_management'] = comfy_mm
    if 'nodes' not in sys.modules:
        nodes = types.ModuleType('nodes')
        nodes.MAX_RESOLUTION = 16384
        sys.modules['nodes'] = nodes


_install_comfy_stubs()

import torch  # noqa: E402
import inpaint_cropandstitch as ics  # noqa: E402


def make_stitcher(H, W, canvas_channels=3, device_mode="cpu (compatible)"):
    """Build a minimal single-region stitcher dict equivalent to what
    InpaintCropImproved emits for a whole-image, identity crop."""
    canvas = torch.rand((1, H, W, canvas_channels))
    mask = torch.ones((1, H, W))  # fully inpaint -> output equals inpainted RGB
    return {
        'downscale_algorithm': 'bilinear',
        'upscale_algorithm': 'bicubic',
        'blend_pixels': 0.0,
        'canvas_to_orig_x': [0], 'canvas_to_orig_y': [0],
        'canvas_to_orig_w': [W], 'canvas_to_orig_h': [H],
        'canvas_image': [canvas],
        'cropped_to_canvas_x': [0], 'cropped_to_canvas_y': [0],
        'cropped_to_canvas_w': [W], 'cropped_to_canvas_h': [H],
        'cropped_mask_for_blend': [mask],
        'device_mode': device_mode,
    }, canvas, mask


def run_case(name, accumulate, color_match, inpaint_channels=4, H=16, W=16):
    node = ics.InpaintStitchImproved()
    stitcher, canvas, _mask = make_stitcher(H, W, canvas_channels=3)
    inpainted = torch.rand((1, H, W, inpaint_channels))
    (out,) = node._stitch_one_call(
        stitcher, inpainted, accumulate=accumulate, color_match=color_match
    )
    assert out.shape[-1] == 3, f"{name}: expected 3 output channels, got {out.shape[-1]}"
    assert out.shape[1] == H and out.shape[2] == W, f"{name}: wrong spatial size {tuple(out.shape)}"
    # mask is all-ones, so the masked-in result should equal the inpainted RGB
    # (within 8-bit resize round-trip tolerance). color_match remaps the
    # distribution, so only assert pixel equality when it is off.
    if not color_match:
        assert torch.allclose(out, inpainted[..., :3], atol=0.02), \
            f"{name}: output does not match inpainted RGB content"
    print(f"  PASS: {name} -> output shape {tuple(out.shape)}")


def main():
    print("Testing RGBA(4ch) inpaint into RGB(3ch) canvas:")
    run_case("basic stitch (accumulate=False, color_match=False)", False, False)
    run_case("color_match path (accumulate=False, color_match=True)", False, True)
    run_case("accumulate path (accumulate=True, color_match=False)", True, False)
    run_case("accumulate + color_match (accumulate=True, color_match=True)", True, True)

    # Sanity: the ordinary 3ch->3ch case must still work unchanged.
    print("Sanity: RGB(3ch) inpaint into RGB(3ch) canvas:")
    run_case("rgb passthrough", False, False, inpaint_channels=3)

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
