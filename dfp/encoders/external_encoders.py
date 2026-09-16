"""External encoder adapters: Java, .NET, Node.js, libarchive.

Each adapter shells out to a real, independently-implemented DEFLATE encoder so
the corpus contains genuinely distinct families, not just zlib re-parameterised.
Helper programs are written once into a cache directory and reused.

Availability is probed lazily and cached; an adapter that cannot run on this
machine simply never enters the corpus.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import Encoder, EncoderResult, register

_CACHE = Path(os.environ.get("KIROCREW_SCRATCH", tempfile.gettempdir())) / "dfp_encoders"
_CACHE.mkdir(parents=True, exist_ok=True)


def _run(argv: list[str], stdin: bytes, timeout: int = 120) -> bytes:
    proc = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{argv[0]} exit {proc.returncode}: {proc.stderr.decode('utf-8','replace')[:400]}"
        )
    return proc.stdout


# --- Java ------------------------------------------------------------------

_JAVA_SRC = r'''
import java.io.*;
import java.util.zip.Deflater;
public class DfpDeflate {
    public static void main(String[] a) throws Exception {
        int level = a.length > 0 ? Integer.parseInt(a[0]) : 6;
        ByteArrayOutputStream in = new ByteArrayOutputStream();
        byte[] buf = new byte[65536]; int r;
        while ((r = System.in.read(buf)) > 0) in.write(buf, 0, r);
        byte[] data = in.toByteArray();
        Deflater d = new Deflater(level, true); // nowrap = raw DEFLATE
        d.setInput(data); d.finish();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] o = new byte[65536];
        while (!d.finished()) { int n = d.deflate(o); out.write(o, 0, n); }
        d.end();
        System.out.write(out.toByteArray());
        System.out.flush();
    }
}
'''


class JavaEncoder(Encoder):
    family = "java"

    def __init__(self) -> None:
        self._ready: bool | None = None
        self._classdir = _CACHE / "java"

    def _compile(self) -> bool:
        if not (shutil.which("javac") and shutil.which("java")):
            return False
        self._classdir.mkdir(exist_ok=True)
        src = self._classdir / "DfpDeflate.java"
        cls = self._classdir / "DfpDeflate.class"
        if not cls.exists():
            src.write_text(_JAVA_SRC, encoding="utf-8")
            try:
                subprocess.run(
                    ["javac", str(src)], cwd=self._classdir, capture_output=True,
                    check=True, timeout=120,
                )
            except Exception:
                return False
        return cls.exists()

    def available(self) -> bool:
        if self._ready is None:
            self._ready = self._compile()
        return self._ready

    def levels(self) -> list[str]:
        return ["1", "6", "9"]

    def compress(self, data: bytes, level: str) -> EncoderResult:
        raw = _run(["java", "-cp", str(self._classdir), "DfpDeflate", level], data)
        return EncoderResult(raw, self.family, level, self.label(level))


# --- .NET ------------------------------------------------------------------

_DOTNET_PROG = r'''
using System;
using System.IO;
using System.IO.Compression;
class Program {
    static void Main(string[] args) {
        var level = args.Length > 0 ? args[0] : "optimal";
        CompressionLevel cl = level switch {
            "fastest" => CompressionLevel.Fastest,
            "nocompress" => CompressionLevel.NoCompression,
            _ => CompressionLevel.Optimal,
        };
        using var stdin = Console.OpenStandardInput();
        using var mem = new MemoryStream();
        stdin.CopyTo(mem);
        var data = mem.ToArray();
        using var outMem = new MemoryStream();
        using (var ds = new DeflateStream(outMem, cl, true)) {
            ds.Write(data, 0, data.Length);
        }
        using var stdout = Console.OpenStandardOutput();
        var res = outMem.ToArray();
        stdout.Write(res, 0, res.Length);
    }
}
'''

_DOTNET_CSPROJ = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net5.0</TargetFramework>
    <Nullable>disable</Nullable>
    <ImplicitUsings>disable</ImplicitUsings>
    <AssemblyName>dfpdotnet</AssemblyName>
  </PropertyGroup>
</Project>
"""


class DotnetEncoder(Encoder):
    family = "dotnet"

    def __init__(self) -> None:
        self._ready: bool | None = None
        self._proj = _CACHE / "dotnet"
        self._dll: Path | None = None

    def _build(self) -> bool:
        if not shutil.which("dotnet"):
            return False
        self._proj.mkdir(exist_ok=True)
        (self._proj / "Program.cs").write_text(_DOTNET_PROG, encoding="utf-8")
        (self._proj / "dfpdotnet.csproj").write_text(_DOTNET_CSPROJ, encoding="utf-8")
        dll = self._proj / "bin" / "Release" / "net5.0" / "dfpdotnet.dll"
        if not dll.exists():
            try:
                subprocess.run(
                    ["dotnet", "build", "-c", "Release", "--nologo", "-v", "q"],
                    cwd=self._proj, capture_output=True, check=True, timeout=400,
                    env={**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
                         "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1"},
                )
            except Exception:
                return False
        if dll.exists():
            self._dll = dll
            return True
        return False

    def available(self) -> bool:
        if self._ready is None:
            self._ready = self._build()
        return self._ready

    def levels(self) -> list[str]:
        return ["fastest", "optimal"]

    def compress(self, data: bytes, level: str) -> EncoderResult:
        raw = _run(["dotnet", str(self._dll), level], data)
        return EncoderResult(raw, self.family, level, self.label(level))


# --- Node.js ---------------------------------------------------------------

_NODE_SRC = r"""
const zlib = require('zlib');
const chunks = [];
process.stdin.on('data', c => chunks.push(c));
process.stdin.on('end', () => {
  const data = Buffer.concat(chunks);
  const level = parseInt(process.argv[2] || '6', 10);
  const out = zlib.deflateRawSync(data, { level });
  process.stdout.write(out);
});
"""


class NodeEncoder(Encoder):
    family = "node"

    def __init__(self) -> None:
        self._ready: bool | None = None
        self._script = _CACHE / "node_deflate.js"

    def available(self) -> bool:
        if self._ready is None:
            if not shutil.which("node"):
                self._ready = False
            else:
                self._script.write_text(_NODE_SRC, encoding="utf-8")
                self._ready = True
        return self._ready

    def levels(self) -> list[str]:
        return ["1", "6", "9"]

    def compress(self, data: bytes, level: str) -> EncoderResult:
        raw = _run(["node", str(self._script), level], data)
        return EncoderResult(raw, self.family, level, self.label(level))


# --- libarchive (bsdtar) ---------------------------------------------------


class LibarchiveEncoder(Encoder):
    """Compress via bsdtar's zip writer, then pull the raw DEFLATE back out.

    libarchive links its own zlib build; the family is worth having because its
    block-splitting and level mapping differ from CPython's in practice.
    """

    family = "libarchive"

    def __init__(self) -> None:
        self._ready: bool | None = None
        self._bin = shutil.which("bsdtar") or shutil.which("tar")

    def available(self) -> bool:
        if self._ready is None:
            if not self._bin:
                self._ready = False
            else:
                try:
                    out = subprocess.run(
                        [self._bin, "--version"], capture_output=True, text=True,
                        timeout=20,
                    )
                    self._ready = "libarchive" in (out.stdout + out.stderr)
                except Exception:
                    self._ready = False
        return self._ready

    def levels(self) -> list[str]:
        return ["default"]

    def compress(self, data: bytes, level: str) -> EncoderResult:
        from ..containers import extract_zip

        with tempfile.TemporaryDirectory(dir=str(_CACHE)) as td:
            tdp = Path(td)
            payload = tdp / "payload.bin"
            payload.write_bytes(data)
            archive = tdp / "out.zip"
            subprocess.run(
                [self._bin, "-a", "-c", "-f", str(archive),
                 "--options", "zip:compression=deflate",
                 "-C", str(tdp), "payload.bin"],
                capture_output=True, check=True, timeout=120,
            )
            report = extract_zip(archive.read_bytes(), str(archive), "zip")
            if not report.streams:
                raise RuntimeError("libarchive produced no DEFLATE stream (stored?)")
            raw = report.streams[0].payload
        return EncoderResult(raw, self.family, level, self.label(level))


register(JavaEncoder())
register(DotnetEncoder())
register(NodeEncoder())
register(LibarchiveEncoder())
