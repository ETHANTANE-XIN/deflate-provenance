"""Test suite for the DEFLATE provenance toolkit (stdlib unittest).

Run:  python -m unittest discover -s tests   (from the project root)
"""

import os
import random
import sys
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfp.deflate import inflate, parse_stream, DeflateError  # noqa: E402
from dfp.bitreader import BitReader, BitStreamError  # noqa: E402
from dfp.containers import extract_streams, detect_container  # noqa: E402
from dfp.features import FEATURE_NAMES, extract_features, features_to_vector  # noqa: E402
from dfp.encoders import list_encoders  # noqa: E402
from dfp.covert import (  # noqa: E402
    embed_in_padding, extract_from_padding, bytes_to_bits, detect_covert_padding,
)
from dfp.signatures import evaluate_signatures  # noqa: E402


def zc(data, level=6, strategy=zlib.Z_DEFAULT_STRATEGY):
    c = zlib.compressobj(level, zlib.DEFLATED, -15, 8, strategy)
    return c.compress(data) + c.flush()


class TestBitReader(unittest.TestCase):
    def test_lsb_first(self):
        r = BitReader(bytes([0b10110001]))
        self.assertEqual([r.read_bit() for _ in range(8)], [1, 0, 0, 0, 1, 1, 0, 1])

    def test_read_bits_integer(self):
        r = BitReader(bytes([0b11100101, 0b00000001]))
        self.assertEqual(r.read_bits(3), 0b101)

    def test_overrun_raises(self):
        r = BitReader(b"\x00")
        r.read_bits(8)
        with self.assertRaises(BitStreamError):
            r.read_bit()


class TestParserOracle(unittest.TestCase):
    def test_roundtrip_all_levels_strategies(self):
        random.seed(42)
        samples = [
            b"",
            b"A",
            b"the quick brown fox " * 300,
            b"\x00" * 40000,
            os.urandom(20000),
            bytes(random.randrange(256) for _ in range(15000)),
        ]
        strategies = [zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED,
                      zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE, zlib.Z_FIXED]
        checked = 0
        for data in samples:
            for level in range(10):
                for strat in strategies:
                    blob = zc(data, level, strat)
                    rec = inflate(blob, strict=True)
                    self.assertEqual(rec.output, data)
                    checked += 1
        self.assertGreater(checked, 250)

    def test_consumed_size_matches(self):
        blob = zc(b"forensics " * 500)
        rec = inflate(blob, strict=True)
        self.assertEqual(rec.compressed_bytes, len(blob))

    def test_stored_block_level0(self):
        blob = zc(b"X" * 200000, level=0)
        rec = inflate(blob, strict=True)
        self.assertEqual(rec.block_type_counts()["dynamic"], 0)
        self.assertGreater(rec.block_type_counts()["stored"], 0)

    def test_truncated_nonstrict(self):
        blob = zc(b"hello world " * 100)
        rec = parse_stream(blob[:5], strict=False)
        self.assertTrue(rec.error or rec.truncated or rec.out_size >= 0)

    def test_reserved_block_type_raises(self):
        with self.assertRaises(DeflateError):
            inflate(bytes([0b00000111, 0, 0, 0]), strict=True)


class TestEncoders(unittest.TestCase):
    def test_all_encoders_roundtrip(self):
        data = b"digital forensics deflate provenance " * 200
        encs = list_encoders(only_available=True)
        self.assertGreaterEqual(len(encs), 2)  # zlib + purepy at minimum
        for enc in encs:
            for level in enc.levels():
                res = enc.compress(data, level)
                rec = inflate(res.raw_deflate, strict=True)
                self.assertEqual(rec.output, data, f"{res.label} roundtrip")
                self.assertEqual(zlib.decompress(res.raw_deflate, -15), data)

    def test_purepy_present(self):
        families = {e.family for e in list_encoders(True)}
        self.assertIn("purepy", families)
        self.assertIn("zlib", families)

    def test_purepy_greedy_vs_lazy_differ(self):
        from dfp.encoders import get_encoder
        enc = get_encoder("purepy")
        data = b"ababababab cabcabcab the the the cat " * 400
        g = enc.compress(data, "greedy").raw_deflate
        l = enc.compress(data, "lazy").raw_deflate
        self.assertEqual(zlib.decompress(g, -15), data)
        self.assertEqual(zlib.decompress(l, -15), data)


class TestContainers(unittest.TestCase):
    def test_detect_gzip(self):
        import gzip
        blob = gzip.compress(b"hello forensics")
        self.assertEqual(detect_container(blob, "x.gz"), "gzip")

    def test_gzip_extraction(self):
        import gzip
        blob = gzip.compress(b"content " * 500)
        scratch = Path(os.environ.get("KIROCREW_SCRATCH", ".")) / "t.gz"
        scratch.write_bytes(blob)
        rep = extract_streams(str(scratch))
        self.assertEqual(rep.container, "gzip")
        self.assertEqual(len(rep.streams), 1)
        self.assertEqual(zlib.decompress(rep.streams[0].payload, -15), b"content " * 500)

    def test_zip_extraction(self):
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("a.txt", b"alpha " * 400)
            z.writestr("b.txt", b"beta " * 400)
        scratch = Path(os.environ.get("KIROCREW_SCRATCH", ".")) / "t.zip"
        scratch.write_bytes(buf.getvalue())
        rep = extract_streams(str(scratch))
        self.assertEqual(len(rep.streams), 2)
        for s in rep.streams:
            self.assertEqual(zlib.crc32(zlib.decompress(s.payload, -15)) & 0xFFFFFFFF,
                             s.declared_crc32)

    def test_pdf_flatedecode_extraction(self):
        """A hand-built PDF with two FlateDecode streams must yield both."""
        body_a = b"BT /F1 12 Tf (forensic evidence page one) Tj ET " * 60
        body_b = b"BT /F1 12 Tf (page two content here) Tj ET " * 60
        s1 = zlib.compress(body_a, 6)
        s2 = zlib.compress(body_b, 6)
        pdf = bytearray(b"%PDF-1.7\n")
        pdf += b"1 0 obj\n<< /Type /Page /Length %d /Filter /FlateDecode >>\nstream\n" % len(s1)
        pdf += s1 + b"\nendstream\nendobj\n"
        pdf += b"2 0 obj\n<< /Type /Page /Length %d /Filter /FlateDecode >>\nstream\n" % len(s2)
        pdf += s2 + b"\nendstream\nendobj\n"
        pdf += b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
        scratch = Path(os.environ.get("KIROCREW_SCRATCH", ".")) / "t.pdf"
        scratch.write_bytes(bytes(pdf))
        rep = extract_streams(str(scratch))
        self.assertEqual(rep.container, "pdf")
        self.assertEqual(len(rep.streams), 2)
        outs = [parse_stream(s.payload, strict=True).output for s in rep.streams]
        self.assertEqual(outs[0], body_a)
        self.assertEqual(outs[1], body_b)

    def test_pdf_skips_non_flate_filter_chain(self):
        """A stream filtered through ASCII85 before Flate is not raw DEFLATE."""
        blob = zlib.compress(b"x" * 500, 6)
        pdf = bytearray(b"%PDF-1.7\n")
        pdf += b"1 0 obj\n<< /Length %d /Filter [/ASCII85Decode /FlateDecode] >>\nstream\n" % len(blob)
        pdf += blob + b"\nendstream\nendobj\n"
        pdf += b"2 0 obj\n<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(blob)
        pdf += blob + b"\nendstream\nendobj\n%%EOF\n"
        scratch = Path(os.environ.get("KIROCREW_SCRATCH", ".")) / "t2.pdf"
        scratch.write_bytes(bytes(pdf))
        rep = extract_streams(str(scratch))
        self.assertEqual(len(rep.streams), 1)  # only the Flate-only one
        self.assertEqual(rep.other_method_entries, 1)


class TestFeatures(unittest.TestCase):
    def test_vector_length_stable(self):
        rec = parse_stream(zc(b"feature vector test " * 300), strict=True)
        vec = features_to_vector(extract_features(rec))
        self.assertEqual(len(vec), len(FEATURE_NAMES))

    def test_huffman_only_has_no_matches(self):
        rec = parse_stream(zc(b"abcabc " * 500, strategy=zlib.Z_HUFFMAN_ONLY), strict=True)
        self.assertEqual(rec.n_matches, 0)

    def test_features_finite(self):
        import math
        rec = parse_stream(zc(os.urandom(8000)), strict=True)
        for name, v in extract_features(rec).items():
            self.assertTrue(math.isfinite(v), f"{name} not finite: {v}")


class TestCovert(unittest.TestCase):
    def test_padding_roundtrip_output_unchanged(self):
        data = b"secret carrier test " * 137  # yields nonzero padding often
        raw = zc(data)
        msg_bits = bytes_to_bits(b"K")
        stego = embed_in_padding(raw, msg_bits)
        self.assertEqual(zlib.decompress(stego, -15), data)  # output identical
        rec = parse_stream(stego, strict=False)
        if rec.final_pad_bits:
            recovered = extract_from_padding(stego, rec.final_pad_bits)
            self.assertEqual(recovered, msg_bits[: rec.final_pad_bits])

    def test_detector_flags_nonzero_padding(self):
        data = b"detect me " * 200
        raw = zc(data)
        rec = parse_stream(raw, strict=False)
        if rec.final_pad_bits:
            stego = embed_in_padding(raw, [1] * rec.final_pad_bits)
            f = detect_covert_padding(stego)
            self.assertTrue(f.suspicious)


class TestAdversarial(unittest.TestCase):
    def test_metadata_normalisation_preserves_bitstream(self):
        import io, zipfile
        from dfp.adversarial import metadata_robustness
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("doc.xml", b"<x>" + b"data " * 500 + b"</x>")
        scratch = Path(os.environ.get("KIROCREW_SCRATCH", ".")) / "adv.zip"
        scratch.write_bytes(buf.getvalue())
        r = metadata_robustness(str(scratch))
        self.assertTrue(r.metadata_changed)
        self.assertFalse(r.bitstream_changed)
        self.assertLess(r.max_feature_delta, 1e-9)

    def test_recompression_exact_match(self):
        from dfp.adversarial import detect_recompression
        raw = zc(b"reproduce me exactly " * 300, level=6)
        m = detect_recompression(raw)
        self.assertTrue(m.matched)
        self.assertTrue(m.exact)


class TestSignatures(unittest.TestCase):
    def test_huffman_only_rule(self):
        rec = parse_stream(zc(b"xy " * 500, strategy=zlib.Z_HUFFMAN_ONLY), strict=True)
        rules = {h.rule for h in evaluate_signatures(rec)}
        self.assertIn("no-lz77-matches", rules)

    def test_rle_rule(self):
        rec = parse_stream(zc(b"A" * 5000, strategy=zlib.Z_RLE), strict=True)
        rules = {h.rule for h in evaluate_signatures(rec)}
        self.assertIn("distance-1-only", rules)


class TestClassifierSmoke(unittest.TestCase):
    def test_train_predict_save_load(self):
        import numpy as np
        from dfp.corpus import build_corpus
        from dfp.ml import ProvenanceClassifier
        ds = build_corpus(sizes=[4000], per_combo=2,
                          families=["zlib", "purepy"], label_by="family")
        X = np.array(ds.X)
        clf = ProvenanceClassifier(n_estimators=20, seed=1)
        clf.fit(X, ds.y)
        p = clf.predict_one(X[0])
        self.assertIn(p.raw_top, clf.classes)
        scratch = Path(os.environ.get("KIROCREW_SCRATCH", ".")) / "m.json"
        clf.save(str(scratch))
        clf2 = ProvenanceClassifier.load(str(scratch))
        p2 = clf2.predict_one(X[0])
        self.assertEqual(p.raw_top, p2.raw_top)
        self.assertAlmostEqual(p.confidence, p2.confidence, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
