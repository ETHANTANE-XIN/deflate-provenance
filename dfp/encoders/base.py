"""Encoder adapter base class and registry.

Terminology (proposal section I and III.B):

``program``  the software that was run, e.g. CPython's ``zlib`` module, Java's
             ``java.util.zip``, Node.js, Go, 7-Zip.  Each adapter has one.
``library``  the DEFLATE implementation the program is built on.  Programs
             that call the same library can share a profile: Java is built on
             zlib, so a JAR written by Java tools and a DOCX written by Python
             share the zlib profile.  The corpus builder *verifies* that claim
             (byte-identical output) before merging, see :mod:`dfp.corpus`.
``setting``  the parameter the program was run with (a level, a strategy).
``profile``  library + version + setting, e.g. "zlib 1.3 at level 6".

An adapter never enters the corpus unless :meth:`Encoder.available` is True,
so the corpus is honest about what was really used to build it, and every
adapter reports the exact version that produced its output.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EncoderResult:
    """A raw DEFLATE stream produced by an encoder, with its ground truth."""

    raw_deflate: bytes
    program: str
    library: str
    setting: str
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.program}/{self.setting}"


class Encoder:
    """Base adapter: subclasses compress to a *raw* DEFLATE stream."""

    #: unique adapter name (the program)
    name: str = "base"
    #: DEFLATE library the program is built on (the candidate profile family)
    library: str = "base"
    #: True for the adapter that defines its library's reference output
    reference: bool = True
    #: True for encoders written by the team (never a real-world profile);
    #: they are used for the unknown-encoder test, not as training classes
    synthetic: bool = False

    def available(self) -> bool:  # pragma: no cover - overridden
        return False

    def settings(self) -> list[str]:  # pragma: no cover - overridden
        return []

    def version(self) -> str:  # pragma: no cover - overridden
        return "unknown"

    def compress(self, data: bytes, setting: str) -> EncoderResult:  # pragma: no cover
        raise NotImplementedError

    def compress_many(self, data: bytes, settings: list[str]) -> dict[str, bytes]:
        """Compress ``data`` at several settings (one process call if possible)."""
        return {s: self.compress(data, s).raw_deflate for s in settings}

    def result(self, raw: bytes, setting: str, **extra) -> EncoderResult:
        return EncoderResult(raw, self.name, self.library, setting, dict(extra))

    def label(self, setting: str) -> str:
        return f"{self.name}/{setting}"

    def describe(self) -> dict:
        return {
            "program": self.name,
            "library": self.library,
            "reference": self.reference,
            "synthetic": self.synthetic,
            "settings": self.settings(),
            "version": self.version(),
        }


_REGISTRY: dict[str, Encoder] = {}


def register(encoder: Encoder) -> None:
    _REGISTRY[encoder.name] = encoder


def get_encoder(name: str) -> Encoder:
    _ensure_loaded()
    return _REGISTRY[name]


def list_encoders(only_available: bool = True, include_synthetic: bool = True) -> list[Encoder]:
    _ensure_loaded()
    encoders = list(_REGISTRY.values())
    if not include_synthetic:
        encoders = [e for e in encoders if not e.synthetic]
    if only_available:
        encoders = [e for e in encoders if e.available()]
    return encoders


def reference_for(library: str) -> Encoder | None:
    """The available reference adapter of ``library``, if any."""
    for enc in list_encoders(only_available=True):
        if enc.library == library and enc.reference:
            return enc
    return None


_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    # Importing each module registers its encoders.
    from . import zlib_encoder  # noqa: F401
    from . import native_encoders  # noqa: F401
    from . import external_encoders  # noqa: F401
    from . import purepy_encoder  # noqa: F401
