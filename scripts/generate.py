import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm

from animations import load_animation, apply_animation
from bounding_box import generate_hitbox, append_box_nodes, get_volume
from config import load_config
from merge import merge_models
from texture import generate_image, resize_tiled_texture


def load_model(path: Path) -> Any:
    model = json.loads(path.read_text(encoding="utf-8"))
    return model


def resize_model(nodes: dict) -> None:
    for node in nodes:
        node["position"] = {
            "x": node["position"]["x"] / 2,
            "y": node["position"]["y"] / 2,
            "z": node["position"]["z"] / 2,
        }
        if "offset" in node["shape"]:
            node["shape"]["offset"] = {
                "x": node["shape"]["offset"]["x"] / 2,
                "y": node["shape"]["offset"]["y"] / 2,
                "z": node["shape"]["offset"]["z"] / 2,
            }
        if "size" in node["shape"]["settings"]:
            node["shape"]["settings"]["size"] = {
                k: v // 2 for k, v in node["shape"]["settings"]["size"].items()
            }
        if "textureLayout" in node["shape"]:
            for face, layout in node["shape"]["textureLayout"].items():
                layout["offset"] = {
                    "x": layout["offset"]["x"] // 2,
                    "y": layout["offset"]["y"] // 2,
                }
        resize_model(node.get("children", []))


def offset_model(nodes: dict, offset_x: int, offset_y: int, offset_z: int) -> None:
    for node in nodes:
        node["position"] = {
            "x": node["position"]["x"] + offset_x,
            "y": node["position"]["y"] + offset_y,
            "z": node["position"]["z"] + offset_z,
        }


def filter_nodes(nodes: list) -> list:
    return [
        {**node, "children": filter_nodes(node.get("children", []))}
        for node in nodes
        if node["shape"]["type"] != "quad"
    ]


def read_language_file(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    lang_dict = {}
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            lang_dict[key.strip()] = value.strip()
    return lang_dict


def main(debug: bool = False, filter_quads: bool = False, clear: bool = False):
    hytale_root = Path(__file__).parent.parent.parent / "Assets/Common"
    assets_root = Path(__file__).parent / "assets"
    pack_root = Path(__file__).parent.parent / "src/main/resources"
    config_path = Path(__file__).with_name("config.json")

    config = load_config(config_path)
    socket_configs = {c.name: c for c in config.sockets}

    # Delete old files
    if clear:
        shutil.rmtree(
            pack_root / "Server/Item/Block/Hitboxes/Furniture", ignore_errors=True
        )
        shutil.rmtree(pack_root / "Server/Item/Items/Statue", ignore_errors=True)
        shutil.rmtree(pack_root / "Common/Blocks/Furniture", ignore_errors=True)
        shutil.rmtree(
            pack_root / "Common/Icons/ItemsGenerated/Statue", ignore_errors=True
        )

    # Make dirs
    (pack_root / "Common/Icons/ItemsGenerated/Statue").mkdir(
        parents=True, exist_ok=True
    )

    # Generate lang file
    language_path = pack_root / "Server/Languages/en-US/items.lang"
    language = read_language_file(pack_root / language_path)

    for model_config in tqdm(config.models, desc="Models"):
        for style_config in config.styles:
            socket_config = (
                socket_configs[model_config.socket] if model_config.socket else None
            )

            model = load_model(hytale_root / model_config.model)

            animation = load_animation(
                hytale_root / model_config.animation, model_config.time
            )
            apply_animation(model["nodes"], animation)
            resize_model(model["nodes"])

            if filter_quads:
                model["nodes"] = filter_nodes(model["nodes"])

            name = f"Ymmersive_Statues_{style_config.name}_{model_config.name}"

            # Offset by multi block
            offset_x = (
                model_config.offset_x + socket_config.model_offset_x
                if socket_config
                else 0
            )
            offset_y = (
                model_config.offset_y + socket_config.model_offset_y
                if socket_config
                else 0
            )
            offset_z = (
                model_config.offset_z + socket_config.model_offset_z
                if socket_config
                else 0
            )
            offset_model(model["nodes"], offset_x, offset_y, offset_z)

            # Generate texture
            texture = generate_image(model, model_config, style_config, hytale_root)

            # Merge socket
            if socket_config:
                socket = load_model(
                    assets_root
                    / f"models/socket_{style_config.socket_shape}_{socket_config.name}.blockymodel"
                )
                socket_light = Image.open(
                    assets_root
                    / f"models/socket_{style_config.socket_shape}_{socket_config.name}.png"
                )
                socket_base_texture = np.asarray(
                    Image.open(assets_root / style_config.socket_material).convert(
                        "RGBA"
                    )
                )
                socket_texture = resize_tiled_texture(
                    socket_base_texture, socket_light.size
                )
                lightness = np.asarray(socket_light)[
                    :, :, :1
                ] / 127.0 * style_config.socket_contrast + 0.5 * (
                    1 - style_config.socket_contrast
                )
                socket_texture[:, :, :3] = (
                    (socket_texture[:, :, :3] * lightness).clip(0, 255).astype(np.uint8)
                )
                socket_texture = Image.fromarray(
                    socket_texture.clip(0, 255).astype(np.uint8)
                )
                socket_texture = ImageEnhance.Brightness(socket_texture).enhance(
                    1 + style_config.socket_brightness
                )

                model, texture = merge_models(model, socket, texture, socket_texture)

            # Generate hitbox
            boxes = generate_hitbox(model)
            if debug:
                append_box_nodes(model, boxes["Boxes"])

            # Generate costs
            cost = math.ceil((get_volume(boxes) ** 0.5) * style_config.resource_factor)

            # Save
            output_path = pack_root / "Common/Blocks/Furniture"
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / f"{name}.blockymodel").write_text(
                json.dumps(model, indent=2), encoding="utf-8"
            )
            texture.save(output_path / f"{name}.png")

            # Generate hitbox
            hitbox_output_path = (
                pack_root / f"Server/Item/Block/Hitboxes/Furniture/{name}.json"
            )
            hitbox_output_path.parent.mkdir(parents=True, exist_ok=True)
            hitbox_output_path.write_text(json.dumps(boxes, indent=2), encoding="utf-8")

            # Generate item json
            item_output_path = pack_root / f"Server/Item/Items/Statue/{name}.json"
            item_output_path.parent.mkdir(parents=True, exist_ok=True)
            template = (
                (assets_root / "item.json")
                .read_text(encoding="utf-8")
                .replace(
                    '"{x}"', str(-socket_config.model_offset_x if socket_config else 0)
                )
                .replace("{name}", name)
                .replace("{material}", style_config.name)
                .replace("{resource}", style_config.resource)
                .replace("{resourceType}", style_config.resource_type)
                .replace("'{cost}'", str(cost))
            )
            item_output_path.write_text(template, encoding="utf-8")

            # Generate fake icon until proper icons are made
            shutil.copy(
                pack_root
                / f"Common/Icons/ItemsGenerated/Statue_Placeholder_{style_config.name}.png",
                pack_root / f"Common/Icons/ItemsGenerated/Statue/{name}.png",
            )

            # Generate lang entry
            key = name + ".name"
            if key not in language:
                language[key] = (
                    f"{style_config.name} Statue of {model_config.name.replace('_', ' ').title()}"
                )

    # Save language file
    lang_lines = [f"{key} = {value}" for key, value in language.items()]
    (pack_root / language_path).write_text("\n".join(lang_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
