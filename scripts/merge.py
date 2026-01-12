from PIL import Image


def adjust(nodes: list, offset_x: int) -> None:
    for node in nodes:
        for uv in node["shape"]["textureLayout"].values():
            uv["offset"]["x"] += offset_x
        adjust(node.get("children", []), offset_x)


def merge_models(
        model_a: dict, model_b: dict, texture_a: Image.Image, texture_b: Image.Image
) -> tuple[dict, Image.Image]:
    model_a["nodes"].extend(model_b["nodes"])

    h = max(texture_a.height, texture_b.height)
    canvas = Image.new("RGBA", (texture_a.width + texture_b.width, h))
    canvas.paste(texture_a, (0, 0))
    canvas.paste(texture_b, (texture_a.width, 0))

    adjust(model_b["nodes"], texture_a.width)

    return model_a, canvas


def round32(x: int) -> int:
    return (x + 31) // 32 * 32


def fix_size(texture: Image.Image) -> Image.Image:
    canvas = Image.new("RGBA", (round32(texture.width), round32(texture.height)))
    canvas.paste(texture, (0, 0))
    return canvas
