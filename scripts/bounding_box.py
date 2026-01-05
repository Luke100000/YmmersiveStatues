import math


def identity_matrix() -> tuple[tuple[float, float, float], ...]:
    return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def quaternion_to_matrix(
    quat: dict | None,
) -> tuple[tuple[float, float, float], ...]:
    x = float((quat or {}).get("x", 0.0))
    y = float((quat or {}).get("y", 0.0))
    z = float((quat or {}).get("z", 0.0))
    w = float((quat or {}).get("w", 1.0))
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def mat_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def mat_vec(m, v):
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def vec_add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def to_vec(data: dict) -> tuple[float, float, float]:
    return float(data["x"]), float(data["y"]), float(data["z"])


def box_from_shape(
    shape: dict, rotation, translation, name: str, debug: bool = False
) -> dict | None:
    size = shape["settings"]["size"]
    stretch = shape.get("stretch") or {"x": 1.0, "y": 1.0, "z": 1.0}
    offset = to_vec(shape["offset"])
    half = (
        abs(float(size["x"]) * float(stretch["x"])) / 2,
        abs(float(size["y"]) * float(stretch["y"])) / 2,
        abs(float(size["z"]) * float(stretch["z"])) / 2,
    )
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local_corner = (
                    offset[0] + sx * half[0],
                    offset[1] + sy * half[1],
                    offset[2] + sz * half[2],
                )
                corners.append(vec_add(translation, mat_vec(rotation, local_corner)))
    min_x = min(p[0] for p in corners)
    min_y = min(p[1] for p in corners)
    min_z = min(p[2] for p in corners)
    max_x = max(p[0] for p in corners)
    max_y = max(p[1] for p in corners)
    max_z = max(p[2] for p in corners)

    center = (
        (min_x + max_x) / 2,
        (min_y + max_y) / 2,
        (min_z + max_z) / 2,
    )
    axis_lengths = (max_x - min_x, max_y - min_y, max_z - min_z)
    original_lengths = (
        abs(float(size["x"]) * float(stretch["x"])),
        abs(float(size["y"]) * float(stretch["y"])),
        abs(float(size["z"]) * float(stretch["z"])),
    )
    original_volume = original_lengths[0] * original_lengths[1] * original_lengths[2]
    aabb_volume = axis_lengths[0] * axis_lengths[1] * axis_lengths[2]
    if original_volume > 0 and aabb_volume > 0:
        scale = (original_volume / aabb_volume) ** (1 / 3)
    else:
        scale = 1.0
    scaled_half = tuple((axis_lengths[i] * scale) / 2 for i in range(3))

    min_x = center[0] - scaled_half[0]
    max_x = center[0] + scaled_half[0]
    min_y = center[1] - scaled_half[1]
    max_y = center[1] + scaled_half[1]
    min_z = center[2] - scaled_half[2]
    max_z = center[2] + scaled_half[2]

    box: dict = {
        "Min": {"X": min_x / 32 + 0.5, "Y": max(0, min_y / 32), "Z": min_z / 32 + 0.5},
        "Max": {"X": max_x / 32 + 0.5, "Y": max(0, max_y / 32), "Z": max_z / 32 + 0.5},
    }

    if debug:
        box["Name"] = name

    return box


def generate_hitbox(model: dict) -> dict:
    boxes: list[dict] = []

    def traverse(nodes, parent_rotation, parent_translation):
        for node in nodes:
            local_position = to_vec(node["position"])
            local_rotation = quaternion_to_matrix(node["orientation"])
            world_rotation = mat_mul(parent_rotation, local_rotation)
            translation_base = vec_add(
                parent_translation, mat_vec(parent_rotation, local_position)
            )
            shape = node["shape"]
            if shape["type"] == "box":
                box = box_from_shape(
                    shape, world_rotation, translation_base, node["name"]
                )
                width = box["Max"]["X"] - box["Min"]["X"]
                height = box["Max"]["Y"] - box["Min"]["Y"]
                depth = box["Max"]["Z"] - box["Min"]["Z"]
                if width > 0 and height > 0 and depth > 0:
                    boxes.append(box)
            offset_vec = to_vec(shape["offset"])
            child_translation = vec_add(
                translation_base, mat_vec(world_rotation, offset_vec)
            )
            traverse(node.get("children", []), world_rotation, child_translation)

    traverse(model["nodes"], identity_matrix(), (0.0, 0.0, 0.0))
    return {
        "Boxes": boxes,
    }


def get_volume(hitbox: dict) -> int:
    resolution = 32

    voxels: set[tuple[int, int, int]] = set()
    for box in hitbox.get("Boxes", []):
        min_x = math.floor(box["Min"]["X"] * resolution)
        max_x = math.ceil(box["Max"]["X"] * resolution)
        min_y = math.floor(box["Min"]["Y"] * resolution)
        max_y = math.ceil(box["Max"]["Y"] * resolution)
        min_z = math.floor(box["Min"]["Z"] * resolution)
        max_z = math.ceil(box["Max"]["Z"] * resolution)

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                for z in range(min_z, max_z):
                    voxels.add((x, y, z))

    return len(voxels) / (resolution**3)


def append_box_nodes(model: dict, boxes: list[dict]) -> None:
    nodes = model.setdefault("nodes", [])
    for idx, box in enumerate(boxes):
        min_corner = {
            "X": box["Min"]["X"] * 32 - 16,
            "Y": box["Min"]["Y"] * 32 - 16,
            "Z": box["Min"]["Z"] * 32 - 16,
        }
        max_corner = {
            "X": box["Max"]["X"] * 32 - 16,
            "Y": box["Max"]["Y"] * 32 - 16,
            "Z": box["Max"]["Z"] * 32 - 16,
        }
        size_x = max_corner["X"] - min_corner["X"]
        size_y = max_corner["Y"] - min_corner["Y"]
        size_z = max_corner["Z"] - min_corner["Z"]
        empty = {
            "offset": {"x": 0, "y": 0},
            "mirror": {"x": True, "y": True},
            "angle": 0,
        }
        node = {
            "id": f"hitbox_{idx}",
            "name": box.get("Name", f"FinalBox_{idx}"),
            "position": {
                "x": min_corner["X"] + size_x / 2,
                "y": min_corner["Y"] + size_y / 2,
                "z": min_corner["Z"] + size_z / 2,
            },
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "shape": {
                "type": "box",
                "offset": {"x": 0.0, "y": 0.0, "z": 0.0},
                "settings": {
                    "size": {"x": size_x, "y": size_y, "z": size_z},
                },
                "textureLayout": {
                    "top": empty,
                    "bottom": empty,
                    "front": empty,
                    "back": empty,
                    "left": empty,
                    "right": empty,
                },
            },
            "children": [],
        }
        nodes.append(node)


def get_bounding_box(hitbox: dict) -> dict:
    min_x = min(box["Min"]["X"] for box in hitbox["Boxes"])
    min_y = min(box["Min"]["Y"] for box in hitbox["Boxes"])
    min_z = min(box["Min"]["Z"] for box in hitbox["Boxes"])
    max_x = max(box["Max"]["X"] for box in hitbox["Boxes"])
    max_y = max(box["Max"]["Y"] for box in hitbox["Boxes"])
    max_z = max(box["Max"]["Z"] for box in hitbox["Boxes"])

    return {
        "Min": {"X": min_x, "Y": min_y, "Z": min_z},
        "Max": {"X": max_x, "Y": max_y, "Z": max_z},
    }
