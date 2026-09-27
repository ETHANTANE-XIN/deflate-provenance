"""External encoder adapters: Java, Node.js, Go, .NET, 7-Zip and libarchive.

Each adapter runs a real, independently built program so the corpus contains
the genuine encoders.  Helper programs are written once into a cache
directory (``$DFP_CACHE``, default ``<tmp>/dfp_cache``), rebuilt only when
their source changes, and called once per input with every setting at once:
the helper replies with a 4-byte big-endian length before each raw stream.

Availability is probed lazily; an adapter that cannot run on this machine
simply never enters the corpus.  Adapters whose program also writes ZIP
archives natively expose :meth:`write_zip`, which the metadata baseline and
the producer-rewrite test use to obtain genuine container metadata.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import Encoder, EncoderResult, register


def cache_dir() -> Path:
    root = os.environ.get("DFP_CACHE") or str(Path(tempfile.gettempdir()) / "dfp_cache")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run(argv: list[str], stdin: bytes = b"", timeout: int = 300, cwd=None) -> bytes:
    proc = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{argv[0]} exit {proc.returncode}: "
            f"{proc.stderr.decode('utf-8', 'replace')[:400]}"
        )
    return proc.stdout


def _split_frames(blob: bytes, settings: list[str]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    pos = 0
    for s in settings:
        n = int.from_bytes(blob[pos : pos + 4], "big")
        pos += 4
        out[s] = blob[pos : pos + n]
        pos += n
    if pos != len(blob):
        raise RuntimeError("helper output did not match the requested settings")
    return out


def _build_dir(tag: str, source: str) -> Path:
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    path = cache_dir() / f"{tag}-{digest}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class _BatchEncoder(Encoder):
    """Shared plumbing for helpers that speak the length-prefixed protocol."""

    _ready: bool | None = None
    _version: str | None = None
    zip_writer = False

    def _argv(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _prepare(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def available(self) -> bool:
        if self._ready is None:
            try:
                self._ready = self._prepare()
            except Exception:
                self._ready = False
        return self._ready

    def version(self) -> str:
        if self._version is None:
            if not self.available():
                return "not available"
            try:
                self._version = _run(self._argv() + ["--version"]).decode().strip()
            except Exception as exc:  # pragma: no cover - defensive
                self._version = f"unknown ({exc})"
        return self._version

    def _require(self) -> None:
        if not self.available():
            raise RuntimeError(f"encoder '{self.name}' is not available on this machine")

    def compress_many(self, data: bytes, settings: list[str]) -> dict[str, bytes]:
        self._require()
        blob = _run(self._argv() + list(settings), data)
        return _split_frames(blob, list(settings))

    def compress(self, data: bytes, setting: str) -> EncoderResult:
        return self.result(self.compress_many(data, [setting])[setting], setting)

    def write_zip(self, entries: list[tuple[str, bytes]], out_path: str, setting: str) -> None:
        self._require()
        with tempfile.TemporaryDirectory(dir=str(cache_dir())) as td:
            argv = self._argv() + ["zip", str(out_path), setting]
            for i, (name, data) in enumerate(entries):
                p = Path(td) / f"e{i}"
                p.write_bytes(data)
                argv += [name, str(p)]
            _run(argv)


# --- Java ------------------------------------------------------------------

_JAVA_SRC = r"""
import java.io.*;
import java.util.zip.*;
public class DfpDeflate {
    public static void main(String[] a) throws Exception {
        if (a.length > 0 && a[0].equals("--version")) {
            System.out.println("Java " + System.getProperty("java.vendor") + " "
                + System.getProperty("java.version") + " java.util.zip.Deflater");
            return;
        }
        if (a.length > 0 && a[0].equals("zip")) {
            ZipOutputStream z = new ZipOutputStream(new FileOutputStream(a[1]));
            z.setLevel(Integer.parseInt(a[2]));
            for (int i = 3; i + 1 < a.length; i += 2) {
                z.putNextEntry(new ZipEntry(a[i]));
                z.write(java.nio.file.Files.readAllBytes(new File(a[i + 1]).toPath()));
                z.closeEntry();
            }
            z.close();
            return;
        }
        byte[] data = System.in.readAllBytes();
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(System.out));
        for (String s : a) {
            Deflater d = new Deflater(Integer.parseInt(s), true);  // nowrap = raw DEFLATE
            d.setInput(data);
            d.finish();
            ByteArrayOutputStream bo = new ByteArrayOutputStream();
            byte[] o = new byte[65536];
            while (!d.finished()) { int n = d.deflate(o); bo.write(o, 0, n); }
            d.end();
            byte[] r = bo.toByteArray();
            out.writeInt(r.length);
            out.write(r);
        }
        out.flush();
    }
}
"""


class JavaEncoder(_BatchEncoder):
    """``java.util.zip.Deflater``: built on zlib, so it shares the zlib
    profile once the corpus has verified byte-identical output."""

    name = "java"
    library = "zlib"
    reference = False
    zip_writer = True

    def _prepare(self) -> bool:
        if not (shutil.which("javac") and shutil.which("java")):
            return False
        self._dir = _build_dir("java", _JAVA_SRC)
        cls = self._dir / "DfpDeflate.class"
        if not cls.exists():
            (self._dir / "DfpDeflate.java").write_text(_JAVA_SRC, encoding="utf-8")
            _run(["javac", "DfpDeflate.java"], cwd=self._dir, timeout=300)
        return cls.exists()

    def _argv(self) -> list[str]:
        return ["java", "-cp", str(self._dir), "DfpDeflate"]

    def settings(self) -> list[str]:
        return [str(i) for i in range(1, 10)]


# --- Node.js (Chromium's zlib) ---------------------------------------------

_NODE_SRC = r"""
const zlib = require('zlib');
if (process.argv[2] === '--version') {
  console.log('Node.js ' + process.version + ' zlib ' + process.versions.zlib);
  process.exit(0);
}
const chunks = [];
process.stdin.on('data', c => chunks.push(c));
process.stdin.on('end', () => {
  const data = Buffer.concat(chunks);
  const parts = [];
  for (const a of process.argv.slice(2)) {
    const r = zlib.deflateRawSync(data, { level: parseInt(a, 10) });
    const h = Buffer.alloc(4);
    h.writeUInt32BE(r.length);
    parts.push(h, r);
  }
  process.stdout.write(Buffer.concat(parts));
});
"""


class NodeEncoder(_BatchEncoder):
    """Node.js ``zlib.deflateRawSync``: Node bundles Chromium's zlib fork,
    whose match finder differs from upstream zlib."""

    name = "node"
    library = "chromium-zlib"
    reference = True

    def _prepare(self) -> bool:
        if not shutil.which("node"):
            return False
        self._script = _build_dir("node", _NODE_SRC) / "deflate.js"
        self._script.write_text(_NODE_SRC, encoding="utf-8")
        return True

    def _argv(self) -> list[str]:
        return ["node", str(self._script)]

    def settings(self) -> list[str]:
        return ["1", "3", "6", "9"]


# --- Go compress/flate -------------------------------------------------------

_GO_SRC = r"""package main

import (
	"archive/zip"
	"bytes"
	"compress/flate"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
)

func main() {
	args := os.Args[1:]
	if len(args) > 0 && args[0] == "--version" {
		fmt.Println("Go " + runtime.Version() + " compress/flate")
		return
	}
	if len(args) > 0 && args[0] == "zip" {
		f, err := os.Create(args[1])
		if err != nil {
			os.Exit(2)
		}
		level, _ := strconv.Atoi(args[2])
		w := zip.NewWriter(f)
		w.RegisterCompressor(zip.Deflate, func(out io.Writer) (io.WriteCloser, error) {
			return flate.NewWriter(out, level)
		})
		for i := 3; i+1 < len(args); i += 2 {
			data, err := os.ReadFile(args[i+1])
			if err != nil {
				os.Exit(3)
			}
			e, _ := w.Create(args[i])
			e.Write(data)
		}
		w.Close()
		f.Close()
		return
	}
	data, _ := io.ReadAll(os.Stdin)
	for _, a := range args {
		level, err := strconv.Atoi(a)
		if err != nil {
			os.Exit(4)
		}
		var buf bytes.Buffer
		w, err := flate.NewWriter(&buf, level)
		if err != nil {
			os.Exit(5)
		}
		w.Write(data)
		w.Close()
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], uint32(buf.Len()))
		os.Stdout.Write(hdr[:])
		os.Stdout.Write(buf.Bytes())
	}
}
"""


class GoEncoder(_BatchEncoder):
    """Go's standard-library ``compress/flate`` (an independent encoder)."""

    name = "go"
    library = "go-flate"
    reference = True
    zip_writer = True

    def _prepare(self) -> bool:
        if not shutil.which("go"):
            return False
        self._dir = _build_dir("go", _GO_SRC)
        self._bin = self._dir / "dfpflate"
        if not self._bin.exists():
            (self._dir / "main.go").write_text(_GO_SRC, encoding="utf-8")
            env = {**os.environ, "GO111MODULE": "off",
                   "GOCACHE": str(self._dir / "gocache")}
            subprocess.run(["go", "build", "-o", "dfpflate", "main.go"], cwd=self._dir,
                           capture_output=True, check=True, timeout=600, env=env)
        return self._bin.exists()

    def _argv(self) -> list[str]:
        return [str(self._bin)]

    def settings(self) -> list[str]:
        return ["1", "3", "6", "9"]


# --- .NET System.IO.Compression --------------------------------------------

_DOTNET_SRC = r"""
using System;
using System.IO;
using System.IO.Compression;

class Program {
    static CompressionLevel Level(string s) {
        switch (s) {
            case "fastest": return CompressionLevel.Fastest;
            case "optimal": return CompressionLevel.Optimal;
            case "smallest": return CompressionLevel.SmallestSize;
            case "nocompress": return CompressionLevel.NoCompression;
        }
        throw new ArgumentException(s);
    }

    static int Main(string[] args) {
        if (args.Length > 0 && args[0] == "--version") {
            Console.WriteLine(".NET " + Environment.Version + " System.IO.Compression");
            return 0;
        }
        if (args.Length > 0 && args[0] == "zip") {
            using (var fs = File.Create(args[1]))
            using (var za = new ZipArchive(fs, ZipArchiveMode.Create)) {
                var lv = Level(args[2]);
                for (int i = 3; i + 1 < args.Length; i += 2) {
                    var e = za.CreateEntry(args[i], lv);
                    using (var es = e.Open()) {
                        var b = File.ReadAllBytes(args[i + 1]);
                        es.Write(b, 0, b.Length);
                    }
                }
            }
            return 0;
        }
        var mem = new MemoryStream();
        using (var stdin = Console.OpenStandardInput()) { stdin.CopyTo(mem); }
        var data = mem.ToArray();
        using (var stdout = Console.OpenStandardOutput()) {
            foreach (var a in args) {
                var o = new MemoryStream();
                using (var ds = new DeflateStream(o, Level(a), true)) {
                    ds.Write(data, 0, data.Length);
                }
                var r = o.ToArray();
                var h = BitConverter.GetBytes(r.Length);
                if (BitConverter.IsLittleEndian) Array.Reverse(h);
                stdout.Write(h, 0, 4);
                stdout.Write(r, 0, r.Length);
            }
        }
        return 0;
    }
}
"""

_DOTNET_CSPROJ = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>{tfm}</TargetFramework>
    <Nullable>disable</Nullable>
    <ImplicitUsings>disable</ImplicitUsings>
    <AssemblyName>dfpdotnet</AssemblyName>
  </PropertyGroup>
</Project>
"""


def _dotnet_major() -> int | None:
    """Highest .NET major version that both an SDK and a runtime provide."""
    try:
        sdks = _run(["dotnet", "--list-sdks"]).decode()
        runtimes = _run(["dotnet", "--list-runtimes"]).decode()
    except Exception:
        return None
    sdk_majors = {int(m) for m in re.findall(r"^(\d+)\.", sdks, re.M)}
    rt_majors = {int(m) for m in re.findall(r"Microsoft\.NETCore\.App (\d+)\.", runtimes)}
    both = sdk_majors & rt_majors
    return max(both) if both else None


class DotnetEncoder(_BatchEncoder):
    """.NET ``DeflateStream``.  From .NET 9 the runtime uses zlib-ng, so that
    is the declared library there; older runtimes declare zlib.  Either way
    the corpus checks the claim before merging profiles."""

    name = "dotnet"
    library = "zlib"
    reference = False
    zip_writer = True

    def _prepare(self) -> bool:
        if not shutil.which("dotnet"):
            return False
        major = _dotnet_major()
        if major is None:
            return False
        self.library = "zlib-ng" if major >= 9 else "zlib"
        tfm = f"net{major}.0"
        self._dir = _build_dir(f"dotnet-{tfm}", _DOTNET_SRC)
        dll = self._dir / "bin" / "Release" / tfm / "dfpdotnet.dll"
        if not dll.exists():
            (self._dir / "Program.cs").write_text(_DOTNET_SRC, encoding="utf-8")
            (self._dir / "dfpdotnet.csproj").write_text(
                _DOTNET_CSPROJ.format(tfm=tfm), encoding="utf-8")
            subprocess.run(
                ["dotnet", "build", "-c", "Release", "--nologo", "-v", "q"],
                cwd=self._dir, capture_output=True, check=True, timeout=600,
                env={**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
                     "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1", "DOTNET_NOLOGO": "1"},
            )
        self._dll = dll
        return dll.exists()

    def _argv(self) -> list[str]:
        return ["dotnet", str(self._dll)]

    def settings(self) -> list[str]:
        return ["fastest", "optimal", "smallest"]


# --- 7-Zip -----------------------------------------------------------------


def _seven_zip() -> str | None:
    for name in ("7zz", "7z", "7za"):
        path = shutil.which(name)
        if path:
            return path
    return None


class SevenZipEncoder(Encoder):
    """7-Zip's own DEFLATE encoder (via its GZIP writer, one stream per call)."""

    name = "7zip"
    library = "7zip"
    reference = True
    zip_writer = True
    _version: str | None = None

    def available(self) -> bool:
        return _seven_zip() is not None

    def version(self) -> str:
        if self._version is None:
            exe = _seven_zip()
            if not exe:
                return "not available"
            text = subprocess.run([exe], capture_output=True, text=True).stdout
            m = re.search(r"7-Zip[^\n:]*?\d+\.\d+[^\n:]*", text)
            self._version = m.group(0).strip() if m else "7-Zip (unknown version)"
        return self._version

    def settings(self) -> list[str]:
        return ["mx1", "mx5", "mx9"]

    def compress(self, data: bytes, setting: str) -> EncoderResult:
        from ..containers import extract_gzip

        with tempfile.TemporaryDirectory(dir=str(cache_dir())) as td:
            src = Path(td) / "payload.bin"
            src.write_bytes(data)
            out = Path(td) / "out.gz"
            _run([_seven_zip(), "a", "-tgzip", f"-{setting}", "-bso0", "-bsp0",
                  str(out), str(src)])
            report = extract_gzip(out.read_bytes(), str(out))
            raw = report.streams[0].payload
        return self.result(raw, setting)

    def write_zip(self, entries: list[tuple[str, bytes]], out_path: str, setting: str) -> None:
        with tempfile.TemporaryDirectory(dir=str(cache_dir())) as td:
            names = []
            for name, data in entries:
                p = Path(td) / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                names.append(name)
            _run([_seven_zip(), "a", "-tzip", "-mm=Deflate", f"-{setting}", "-bso0",
                  "-bsp0", str(Path(out_path).resolve()), *names], cwd=td)


# --- libarchive (bsdtar) ---------------------------------------------------


class LibarchiveEncoder(Encoder):
    """bsdtar's ZIP writer.  libarchive links the system zlib, so it declares
    the zlib library; the corpus verifies that before merging."""

    name = "libarchive"
    library = "zlib"
    reference = False
    zip_writer = True
    _version: str | None = None

    def _bin(self) -> str | None:
        return shutil.which("bsdtar")

    def available(self) -> bool:
        exe = self._bin()
        if not exe:
            return False
        try:
            out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                                 timeout=20)
        except Exception:
            return False
        return "libarchive" in (out.stdout + out.stderr)

    def version(self) -> str:
        if self._version is None:
            exe = self._bin()
            self._version = (
                subprocess.run([exe, "--version"], capture_output=True, text=True)
                .stdout.strip() if exe else "not available"
            )
        return self._version

    def settings(self) -> list[str]:
        return ["1", "6", "9"]

    def compress(self, data: bytes, setting: str) -> EncoderResult:
        from ..containers import extract_zip

        with tempfile.TemporaryDirectory(dir=str(cache_dir())) as td:
            archive = Path(td) / "out.zip"
            self.write_zip([("payload.bin", data)], str(archive), setting)
            report = extract_zip(archive.read_bytes(), str(archive), "zip")
            if not report.streams:
                raise RuntimeError("libarchive produced no DEFLATE stream (stored?)")
            raw = report.streams[0].payload
        return self.result(raw, setting)

    def write_zip(self, entries: list[tuple[str, bytes]], out_path: str, setting: str) -> None:
        with tempfile.TemporaryDirectory(dir=str(cache_dir())) as td:
            names = []
            for name, data in entries:
                p = Path(td) / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                names.append(name)
            _run([self._bin(), "-c", "--format", "zip", "-f", str(Path(out_path).resolve()),
                  "--options", f"zip:compression=deflate,zip:compression-level={setting}",
                  *names], cwd=td)


register(JavaEncoder())
register(NodeEncoder())
register(GoEncoder())
register(DotnetEncoder())
register(SevenZipEncoder())
register(LibarchiveEncoder())
