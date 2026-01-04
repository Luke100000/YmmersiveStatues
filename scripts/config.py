import json
from pathlib import Path

from pydantic import BaseModel, Field


class OverlayConfig(BaseModel):
    texture: str
    blendMode: str = "add"
    opacity: float = 1.0


class StyleConfig(BaseModel):
    name: str
    texture: str
    overlays: list[OverlayConfig] = []
    strength: float = 0.15
    adjustment: float = 0.75
    contrast: float = 1.0
    brightness: float = 0.0
    softness: float = 1.0

    resource: str = "Rock"
    resource_type: str = "ResourceTypeId"
    resource_factor: float = 8.0

    socket_shape: str = "stone"
    socket_material: str = "base/mossy.png"
    socket_contrast: float = 0.25
    socket_brightness: float = -0.075


class SocketConfig(BaseModel):
    name: str
    model_offset_x: int
    model_offset_y: int
    model_offset_z: int


class ModelConfig(BaseModel):
    name: str
    model: str
    texture: str
    animation: str
    time: float = Field(0.5, ge=0.0)
    socket: str | None = None
    offset_x: int = 0
    offset_y: int = 0
    offset_z: int = 0
    z_fights: bool = False


class GeneratorConfig(BaseModel):
    sockets: list[SocketConfig]
    styles: list[StyleConfig]
    models: list[ModelConfig]


def load_config(path: Path) -> GeneratorConfig:
    return GeneratorConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
