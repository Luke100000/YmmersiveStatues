import json
from pathlib import Path


def interpolate_position(a: dict, b: dict, t: float) -> dict:
    if not isinstance(a["delta"], dict) or not isinstance(b["delta"], dict):
        return {}
    return {k: a["delta"][k] * (1 - t) + b["delta"][k] * t for k in a["delta"]}


def interpolate_rotation(a: dict, b: dict, t: float) -> dict:
    out = {k: a["delta"][k] * (1 - t) + b["delta"][k] * t for k in a["delta"]}
    n = (out["x"] ** 2 + out["y"] ** 2 + out["z"] ** 2 + out["w"] ** 2) ** 0.5
    return {k: out[k] / n for k in out}


def load_animation(path: Path, time: float) -> dict:
    time *= 60
    animation = json.loads(path.read_text(encoding="utf-8"))

    baked_animation = {}
    for node, animation in animation["nodeAnimations"].items():
        baked_animation[node] = {}
        for value, keyframes in animation.items():
            if keyframes:
                last_keyframe = keyframes[0]
                keyframe = last_keyframe
                for keyframe in keyframes:
                    if keyframe["time"] > time:
                        break
                    last_keyframe = keyframe

                baked_animation[node][value] = (
                    interpolate_rotation
                    if value == "orientation"
                    else interpolate_position
                )(
                    last_keyframe,
                    keyframe,
                    (time - last_keyframe["time"])
                    / max(0.000001, keyframe["time"] - last_keyframe["time"]),
                )

                # baked_animation[node][value] = last_keyframe["delta"]
    return baked_animation


def apply_orientation(node, delta):
    # Normalize delta quaternion
    x, y, z, w = delta["x"], delta["y"], delta["z"], delta["w"]
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n

    # Multiply base orientation by delta
    bx = node["orientation"]["x"]
    by = node["orientation"]["y"]
    bz = node["orientation"]["z"]
    bw = node["orientation"]["w"]

    node["orientation"]["x"] = bw * x + bx * w + by * z - bz * y
    node["orientation"]["y"] = bw * y - bx * z + by * w + bz * x
    node["orientation"]["z"] = bw * z + bx * y - by * x + bz * w
    node["orientation"]["w"] = bw * w - bx * x - by * y - bz * z


def apply_animation(nodes: list[dict], animation: dict) -> None:
    for node in nodes:
        apply_animation(node.get("children", []), animation)

        if node["name"] in animation:
            for value, delta in animation[node["name"]].items():
                if value == "orientation":
                    apply_orientation(node, delta)
                elif value == "shapeStretch":
                    pass
                elif value == "shapeUvOffset":
                    pass
                else:
                    for k, v in delta.items():
                        node[value][k] += v
