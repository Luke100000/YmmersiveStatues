import math
import random
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from config import ModelConfig, StyleConfig
from PIL import Image

TILE_BLEND_MASK = Path(__file__).parent / "assets/mask.png"


def tile_texture(source: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    repeat_y = math.ceil(height / source.shape[0])
    repeat_x = math.ceil(width / source.shape[1])
    full_tiled = np.tile(source, (repeat_y, repeat_x, 1))

    first_tiled = full_tiled[:height, :width, :]
    second_tiled = full_tiled[-height:, -width:, :]

    # Prepare mask
    mask = Image.open(TILE_BLEND_MASK).convert("L")
    mask_resized = mask.resize((width, height), Image.Resampling.BILINEAR)
    mask_arr = np.asarray(mask_resized, dtype=np.float32) / 255.0
    mask_arr = mask_arr[..., np.newaxis]

    # Blend
    tiled_f = first_tiled.astype(np.float32)
    shifted_f = second_tiled.astype(np.float32)
    blended = tiled_f * (1.0 - mask_arr) + shifted_f * mask_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def resize_tiled_texture(source: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    repeat_y = math.ceil(height / source.shape[0])
    repeat_x = math.ceil(width / source.shape[1])
    full_tiled = np.tile(source, (repeat_y, repeat_x, 1))
    return full_tiled[:height, :width, :]


def get_local_contrast(img, sigma_large=1.5, sigma_small=0.25):
    large = ndi.gaussian_filter(img, sigma_large)
    small = ndi.gaussian_filter(img, sigma_small)
    return small - large


def get_face_size(size, side) -> tuple[int, int]:
    if side == "front":
        return size["x"], size["y"]
    elif side == "back":
        return size["x"], size["y"]
    elif side == "left":
        return size["z"], size["y"]
    elif side == "right":
        return size["z"], size["y"]
    elif side == "top":
        return size["x"], size["z"]
    elif side == "bottom":
        return size["x"], size["z"]
    else:
        raise ValueError(f"Unknown side: {side}")


def transform(face, size):
    fw, fh = size
    ox, oy = face["offset"]["x"], face["offset"]["y"]
    angle = face.get("angle", 0) % 360
    mx, my = face["mirror"]["x"], face["mirror"]["y"]

    px = fw if mx else 0
    py = fh if my else 0

    if angle == 0:
        rx, ry = px, py
        bw, bh = fw, fh
    elif angle == 90:
        rx, ry = fh - py, px
        bw, bh = fh, fw
    elif angle == 180:
        rx, ry = fw - px, fh - py
        bw, bh = fw, fh
    elif angle == 270:
        rx, ry = py, fw - px
        bw, bh = fh, fw
    else:
        raise ValueError("unsupported angle")

    x0 = ox - rx
    y0 = oy - ry
    x1 = x0 + bw
    y1 = y0 + bh

    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    return x0, x1, y0, y1


def generate_image(
    model: dict, config: ModelConfig, style: StyleConfig, asset_root: Path
) -> Image.Image:
    # Downscale
    image = np.asarray(Image.open(asset_root / config.texture).convert("RGBA"))
    image = image[1::2, 1::2, :]
    h, w = image.shape[0], image.shape[1]

    # Calculate lightness
    lightness = (
        0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    ) / 255.0

    # Quad-local contrast
    local_contrast = np.zeros((h, w), dtype=np.float32)

    # Prepare background
    texture = Image.open(Path(__file__).parent / "assets" / style.texture).convert(
        "RGBA"
    )
    base_color = np.zeros((h, w, 4), dtype=np.uint8)

    random.seed(42)

    def traverse(
        nodes: list,
        filter_: str,
        tex: Image.Image | None,
        mode: str = "base",
        opacity: float = 1.0,
        seed: int = -1,
    ):
        for node in nodes:
            shape = node["shape"]

            if shape["type"] == filter_:
                if "shadingMode" in shape:
                    del shape["shadingMode"]

                shape["stretch"]["x"] += random.random() * (
                    0.01 if config.z_fights else 0.001
                )
                shape["stretch"]["y"] += random.random() * (
                    0.01 if config.z_fights else 0.001
                )
                shape["stretch"]["z"] += random.random() * (
                    0.01 if config.z_fights else 0.001
                )
                node["position"]["x"] += (random.random() - 0.5) * (
                    0.1 if config.z_fights else 0.01
                )
                node["position"]["y"] += (random.random() - 0.5) * (
                    0.1 if config.z_fights else 0.01
                )
                node["position"]["z"] += (random.random() - 0.5) * (
                    0.1 if config.z_fights else 0.01
                )
                node["orientation"]["w"] += (random.random() - 0.5) * (
                    0.01 if config.z_fights else 0.001
                )
                node["orientation"]["x"] += (random.random() - 0.5) * (
                    0.01 if config.z_fights else 0.001
                )
                node["orientation"]["y"] += (random.random() - 0.5) * (
                    0.01 if config.z_fights else 0.001
                )
                node["orientation"]["z"] += (random.random() - 0.5) * (
                    0.01 if config.z_fights else 0.001
                )

                size = shape["settings"]["size"]
                for side, face in shape["textureLayout"].items():
                    fw, fh = get_face_size(size, side)
                    x0, x1, y0, y1 = transform(face, (fw, fh))

                    # Clamp coordinates to valid range
                    x0 = max(0, x0)
                    x1 = min(w, x1)
                    y0 = max(0, y0)
                    y1 = min(h, y1)

                    if x0 >= x1 or y0 >= y1:
                        continue

                    if tex:
                        source = np.asarray(tex)
                        source = np.roll(
                            source, shift=(x0 + seed * 13, y0 + seed * 7), axis=0
                        )
                        fragment = tile_texture(source, (x1 - x0, y1 - y0))
                        fragment_rgb = fragment[:, :, :3]

                        if mode == "base":
                            base_color[y0:y1, x0:x1, :3] = fragment_rgb
                        elif mode == "add":
                            base_color[y0:y1, x0:x1, :3] += fragment_rgb * opacity
                        elif mode == "multiply":
                            base_color[y0:y1, x0:x1, :3] = np.clip(
                                base_color[y0:y1, x0:x1, :3]
                                * (fragment_rgb / 255.0 * opacity + (1.0 - opacity)),
                                0,
                                255,
                            ).astype(np.uint8)
                        elif mode == "blend":
                            alpha = (fragment[:, :, 3:4] / 255.0) * opacity
                            base_color[y0:y1, x0:x1, :3] = np.clip(
                                base_color[y0:y1, x0:x1, :3] * (1.0 - alpha)
                                + fragment_rgb * alpha,
                                0,
                                255,
                            ).astype(np.uint8)
                        else:
                            raise ValueError(f"Unknown blend mode: {mode}")
                    else:
                        # Calculate local contrast
                        local_contrast[y0:y1, x0:x1] = get_local_contrast(
                            lightness[y0:y1, x0:x1],
                            style.softness * 1.5,
                            style.softness * 0.25,
                        )

            traverse(node.get("children", []), filter_, tex, mode, opacity)

    for type_ in ("box", "quad"):
        traverse(model["nodes"], type_, None)
        traverse(model["nodes"], type_, texture)

        for index, overlay in enumerate(style.overlays):
            traverse(
                model["nodes"],
                type_,
                Image.open(Path(__file__).parent / "assets" / overlay.texture).convert(
                    "RGBA"
                ),
                overlay.blendMode,
                overlay.opacity,
                index,
            )

    lightness += (
        np.pow(np.abs(local_contrast), style.adjustment)
        * np.sign(local_contrast)
        * style.contrast
    )
    lightness -= lightness.mean()
    lightness /= lightness.std()
    lightness += style.brightness

    # Shade
    x = base_color[:, :, :3].astype(np.float32) / 255.0
    eps = 1e-6
    log_x = np.log(x + eps)
    gain = lightness * style.strength
    lit = np.exp(log_x + gain[..., None])
    base_color[:, :, :3] = np.clip(lit * 255, 0, 255).astype(np.uint8)

    base_color[:, :, 3] = (image[:, :, 3] > 100) * 255

    return Image.fromarray(base_color)
