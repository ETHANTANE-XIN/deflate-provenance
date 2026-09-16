"""Label schemes: how specific a provenance claim the bitstream can support.

Vendor names are not always separable, because several tools link the *same*
zlib code base and emit near-identical bytes.  Rather than pretend otherwise,
the tool lets the investigator choose the granularity of the claim:

``fine``     every encoder family kept separate (zlib, java, dotnet, node,
             libarchive, purepy) -- highest resolution, lowest confidence where
             families overlap.
``lineage``  zlib-code-base tools that are genuinely indistinguishable are
             merged (libarchive -> zlib), the rest kept separate.  The default.
``coarse``   collapse every zlib-code-base tool (zlib, Info-ZIP/libarchive,
             Java, Node's zlib build, and .NET's managed zlib port) into one
             ``zlib_lineage`` class, versus ``non_standard`` for an encoder that
             is NOT the mainstream zlib toolchain (our own ``purepy`` is the
             exemplar).  This is the highest-confidence, court-usable claim:
             "produced by the standard zlib toolchain" vs "produced by something
             else".  ``purepy`` is a training class here, not the open-set
             unknown.

Each scheme is a mapping from encoder family to class label; ``purepy`` is kept
separate everywhere so it can serve as the held-out open-set unknown.
"""

from __future__ import annotations

FINE = {
    "zlib": "zlib", "java": "java", "libarchive": "libarchive",
    "node": "node", "dotnet": "dotnet", "purepy": "purepy",
}

LINEAGE = {
    "zlib": "zlib", "java": "java", "libarchive": "zlib",
    "node": "node", "dotnet": "dotnet", "purepy": "purepy",
}

COARSE = {
    "zlib": "zlib_lineage", "java": "zlib_lineage", "libarchive": "zlib_lineage",
    "node": "zlib_lineage", "dotnet": "zlib_lineage", "purepy": "non_standard",
}

SCHEMES = {"fine": FINE, "lineage": LINEAGE, "coarse": COARSE}


def relabel(labels: list[str], scheme: str = "lineage") -> list[str]:
    mapping = SCHEMES[scheme]
    return [mapping.get(l, l) for l in labels]
