"""Encoder adapter base classes and registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EncoderResult:
    """A raw DEFLATE stream produced by an encoder, with its ground-truth label."""

    raw_deflate: bytes
    family: str
    level: str
    label: str
    extra: dict = field(default_factory=dict)


class Encoder:
    """Base adapter: subclasses compress to a *raw* DEFLATE stream."""

    family: str = "base"

    def available(self) -> bool:  # pragma: no cover - overridden
        return False

    def levels(self) -> list[str]:  # pragma: no cover - overridden
        return []

    def compress(self, data: bytes, level: str) -> EncoderResult:  # pragma: no cover
        raise NotImplementedError

    def label(self, level: str) -> str:
        return f"{self.family}/{level}"


_REGISTRY: dict[str, Encoder] = {}


def register(encoder: Encoder) -> None:
    _REGISTRY[encoder.family] = encoder


def get_encoder(family: str) -> Encoder:
    return _REGISTRY[family]


def list_encoders(only_available: bool = True) -> list[Encoder]:
    _ensure_loaded()
    encoders = list(_REGISTRY.values())
    if only_available:
        encoders = [e for e in encoders if e.available()]
    return encoders


_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    # Importing each module registers its encoder.
    from . import zlib_encoder  # noqa: F401
    from . import purepy_encoder  # noqa: F401
    from . import external_encoders  # noqa: F401
