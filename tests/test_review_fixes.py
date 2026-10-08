"""Regression tests for the fixes made after the October 2026 code review.

Each test reproduces one defect the review demonstrated and checks the fix.
Run from the project root:  python -m unittest discover -s tests
"""

import gzip
import io
import os
import random
import sys
import tempfile
import unittest
import warnings
import zipfile
import zlib
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from dfp import deflate as deflate_mod  # noqa: E402
from dfp.aggregate import INSUFFICIENT, analyse_archive  # noqa: E402
from dfp.containers import detect_container, extract_streams  # noqa: E402
from dfp.deflate import inflate, parse_stream  # noqa: E402
from dfp.encoders.zlib_encoder import zlib_raw  # noqa: E402
from dfp.features import FEATURE_NAMES, stream_features  # noqa: E402
from dfp.ml.classifier import UNKNOWN, Prediction  # noqa: E402

TMP = tempfile.TemporaryDirectory()
TMPDIR = Path(TMP.name)
os.environ.setdefault("DFP_CACHE", str(TMPDIR / "cache"))
os.environ["DFP_MODELS"] = str(TMPDIR / "store")


def xml(n, seed=1):
    from dfp.corpus import _xml

    return _xml(random.Random(seed), n)


def text(n, seed=1):
    from dfp.corpus import _text

    return _text(random.Random(seed), n)


def docx(entries, app=None, level=6):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as z:
        if app is not None:
            z.writestr("docProps/app.xml",
                       f"<Properties><Application>{app}</Application></Properties>")
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


def write(name, data):
    path = TMPDIR / name
    path.write_bytes(data)
    return path


class BitWriter:
    """LSB-first bit writer for hand-built DEFLATE streams."""

    def __init__(self):
        self.bits = []

    def write(self, value, n):
        for i in range(n):
            self.bits.append((value >> i) & 1)

    def code(self, value, n):  # Huffman codes are sent MSB-first
        for i in reversed(range(n)):
            self.bits.append((value >> i) & 1)

    def getvalue(self, extra=b""):
        bits = self.bits + [0] * (-len(self.bits) % 8)
        out = bytearray()
        for i in range(0, len(bits), 8):
            out.append(sum(b << k for k, b in enumerate(bits[i:i + 8])))
        return bytes(out) + extra


_OUT = FEATURE_NAMES.index("str_out_size_log")


class StubClassifier:
    """Fixed predictions keyed by an entry's uncompressed size (no training)."""

    def __init__(self, table, classes, strong=0.95, min_evidence=256):
        self.table = table
        self.classes = classes
        self.min_confidence = 0.6
        self.strong_confidence = strong
        self.max_distance = 10.0
        self.min_evidence_bytes = min_evidence
        self.profile_info = {}
        self.novelty_info = {}

    def predict_one(self, vec, explain=5):
        out = int(round(10 ** vec[_OUT] - 1))
        label, conf = self.table.get(out, ("unknown", 0.3))
        abstain = conf < self.min_confidence or label not in self.classes
        return Prediction(UNKNOWN if abstain else label, conf, abstain,
                          "stub" if not abstain else "confidence below 0.60", 1.0,
                          {label: conf}, label, setting=None if abstain else "s")


def kinds(v):
    return {(f.check, f.kind) for f in v.findings}


# --- parser -------------------------------------------------------------------------------


class TestParserStrictness(unittest.TestCase):
    def test_stored_len_nlen_mismatch_is_an_error_in_non_strict_mode(self):
        blob = b"\x01\x03\x00\x00\x00abc"  # final stored block, NLEN not ~LEN
        rec = parse_stream(blob, strict=False)
        self.assertIsNotNone(rec.error)
        self.assertIn("LEN/NLEN", rec.error)
        with self.assertRaises(zlib.error):
            zlib.decompress(blob, -15)

    def test_too_many_length_symbols_rejected_like_zlib(self):
        w = BitWriter()
        w.write(1, 1), w.write(2, 2)          # final dynamic block
        w.write(30, 5), w.write(0, 5), w.write(0, 4)  # HLIT = 287
        blob = w.getvalue(b"\x00" * 8)
        self.assertIn("too many", parse_stream(blob, strict=False).error)

    def test_incomplete_code_length_code_rejected_like_zlib(self):
        w = BitWriter()
        w.write(1, 1), w.write(2, 2)
        w.write(0, 5), w.write(0, 5), w.write(0, 4)  # HLIT 257, HDIST 1, HCLEN 4
        for length in (0, 0, 0, 1):                   # only symbol 0 coded: incomplete
            w.write(length, 3)
        blob = w.getvalue(b"\x00" * 8)
        self.assertIn("incomplete", parse_stream(blob, strict=False).error)
        with self.assertRaises(zlib.error):
            zlib.decompress(blob, -15)

    def test_empty_blocks_do_not_allocate_histograms(self):
        w = BitWriter()
        for i in range(2000):
            w.write(1 if i == 1999 else 0, 1)  # BFINAL on the last one
            w.write(1, 2)                      # fixed codes
            w.code(0, 7)                       # end-of-block
        rec = parse_stream(w.getvalue(), strict=False)
        self.assertIsNone(rec.error)
        self.assertEqual(rec.n_blocks, 2000)
        self.assertTrue(all(b.literal_hist is deflate_mod._ZERO_256 for b in rec.blocks))
        self.assertEqual(zlib.decompress(w.getvalue(), -15), b"")

    def test_analysis_cap_stops_at_a_block_boundary(self):
        data = xml(600000, 9)
        raw = zlib_raw(data, 6)
        rec = parse_stream(raw, max_bytes=8000)
        self.assertTrue(rec.capped)
        self.assertIsNone(rec.error)
        self.assertLess(rec.compressed_bytes, len(raw))
        self.assertEqual(rec.output, data[: rec.out_size])  # an exact prefix
        self.assertFalse(parse_stream(raw).capped)

    def test_capped_entry_votes_with_its_full_size_and_skips_reencoding(self):
        from dfp.aggregate import analyse_stream
        from dfp.containers import DeflateStream

        raw = zlib_raw(xml(600000, 9), 6)
        e = analyse_stream(DeflateStream(payload=raw, name="big"), None, max_bytes=8000)
        self.assertTrue(e.capped)
        self.assertEqual(e.compressed_size, len(raw))
        path = write("big.zip", docx([("big.xml", xml(600000, 9))]))
        with mock.patch("dfp.aggregate.ANALYSE_MAX_BYTES", 8000):
            from dfp import aggregate

            v = aggregate.analyse_archive(str(path), None)
        self.assertTrue(v.entries[0].capped)
        self.assertIsNone(v.entries[0].zlib_matches)

    def test_valid_streams_still_parse_exactly(self):
        for data in (b"", b"x", xml(40000), os.urandom(5000)):
            for lv in (0, 1, 6, 9):
                self.assertEqual(inflate(zlib_raw(data, lv), strict=True).output, data)
            self.assertEqual(inflate(zlib_raw(data, 6, sync_flush=True), strict=True).output, data)


# --- containers ---------------------------------------------------------------------------


class TestContainers(unittest.TestCase):
    def test_multi_member_gzip_every_member_is_analysed(self):
        a, b = text(30000, 1), xml(25000, 2)
        path = write("two.gz", gzip.compress(a) + gzip.compress(b, compresslevel=1))
        rep = extract_streams(str(path))
        self.assertEqual(len(rep.streams), 2)
        self.assertEqual([zlib.decompress(s.payload, -15) for s in rep.streams], [a, b])
        self.assertEqual([s.declared_usize for s in rep.streams], [len(a), len(b)])
        self.assertEqual(rep.metadata["members"], 2)

    def test_gzip_trailing_bytes_reported(self):
        path = write("pad.gz", gzip.compress(text(5000)) + b"\x00" * 512)
        rep = extract_streams(str(path))
        self.assertEqual(len(rep.streams), 1)
        self.assertEqual(rep.metadata["trailing_bytes"], 512)
        self.assertEqual(rep.streams[0].declared_usize, 5000)

    def test_encrypted_entries_are_not_parsed(self):
        data = bytearray(docx([("secret.xml", xml(20000))]))
        local = data.find(b"PK\x03\x04")
        central = data.find(b"PK\x01\x02")
        data[local + 6] |= 1   # general-purpose flag bit 0: encrypted
        data[central + 8] |= 1
        path = write("enc.zip", bytes(data))
        rep = extract_streams(str(path))
        self.assertEqual(rep.streams, [])
        self.assertEqual(rep.encrypted_entries, 1)
        v = analyse_archive(str(path), None)
        self.assertEqual([e.method for e in v.entries], ["encrypted"])
        self.assertEqual(v.entries[0].status, INSUFFICIENT)

    def test_zip64_archives_are_read(self):
        saved = (zipfile.ZIP64_LIMIT, zipfile.ZIP_FILECOUNT_LIMIT)
        zipfile.ZIP64_LIMIT, zipfile.ZIP_FILECOUNT_LIMIT = 50, 2
        try:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for i in range(3):
                    z.writestr(f"part{i}.xml", xml(9000, i))
        finally:
            zipfile.ZIP64_LIMIT, zipfile.ZIP_FILECOUNT_LIMIT = saved
        data = buf.getvalue()
        self.assertIn(b"PK\x06\x06", data)  # ZIP64 end-of-central-directory written
        rep = extract_streams("z64.zip", data=data)
        self.assertTrue(rep.metadata["zip64"])
        self.assertEqual(len(rep.streams), 3)
        for i, s in enumerate(rep.streams):
            self.assertNotEqual(s.declared_csize, 0xFFFFFFFF)
            self.assertEqual(zlib.decompress(s.payload, -15), xml(9000, i))

    def test_unadjusted_prepended_stub_is_rebased(self):
        stub = b"MZ" + b"\x00" * 4094
        path = write("sfx.exe", stub + docx([("a.xml", xml(8000, 1)), ("b.xml", xml(8000, 2))]))
        data = path.read_bytes()
        self.assertEqual(detect_container(data, path), "zip")
        rep = extract_streams(str(path))
        self.assertEqual(rep.metadata["offset_shift"], 4096)
        self.assertEqual([zlib.decompress(s.payload, -15) for s in rep.streams],
                         [xml(8000, 1), xml(8000, 2)])

    def test_backslash_names_still_yield_the_claim(self):
        data = docx([("word/document.xml", xml(5000))], app="Microsoft Office Word")
        data = data.replace(b"docProps/app.xml", b"docProps\\app.xml")
        rep = extract_streams("bs.docx", data=data)
        self.assertEqual(rep.claims["producer"], "Microsoft Office Word")
        self.assertEqual(rep.metadata["backslash_names"], 1)

    def test_jar_manifest_continuation_lines_are_joined(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("META-INF/MANIFEST.MF",
                       "Manifest-Version: 1.0\r\nCreated-By: Apache Maven 3.9.6 with a ver\r\n"
                       " y long build string\r\n")
        rep = extract_streams("m.jar", data=buf.getvalue())
        self.assertEqual(rep.claims["producer"],
                         "Apache Maven 3.9.6 with a very long build string")

    def test_duplicate_names_each_copy_checked_against_its_own_bytes(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            z.writestr("word/document.xml", xml(30000, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # zipfile warns about the duplicate
            with zipfile.ZipFile(buf, "a", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                z.writestr("word/document.xml", xml(30000, 2))
        path = write("dup.docx", buf.getvalue())
        v = analyse_archive(str(path), None)
        self.assertIn(("duplicate-names", "inconsistent"), kinds(v))
        first, second = [e for e in v.entries if e.name == "word/document.xml"]
        self.assertIn(1, {m.level for m in first.zlib_matches})
        self.assertNotIn(1, {m.level for m in second.zlib_matches})
        self.assertIn(9, {m.level for m in second.zlib_matches})


# --- features -----------------------------------------------------------------------------


class TestFeatures(unittest.TestCase):
    def test_flush_blocks_do_not_change_block_splitting_features(self):
        data = xml(40000, 3)
        _, plain = stream_features(zlib_raw(data, 6))
        _, flushed = stream_features(zlib_raw(data, 6, sync_flush=True))
        for name in ("blk_frac_stored", "blk_out_min", "blk_count", "blk_last_is_stored"):
            self.assertEqual(plain[name], flushed[name], name)
        self.assertEqual((plain["blk_flush_markers"], flushed["blk_flush_markers"]), (0.0, 1.0))
        self.assertEqual(flushed["blk_term_empty_coded"], 1.0)
        self.assertEqual(flushed["blk_term_empty_stored"], 0.0)

    def test_distance_over_4k_feature_uses_the_right_codes(self):
        block = os.urandom(3000)
        _, f = stream_features(zlib_raw(block + block, 9))  # matches at distance 3000
        self.assertEqual(f["dis_frac_gt_4k"], 0.0)
        far = os.urandom(5000)
        _, g = stream_features(zlib_raw(far + far, 9))      # distance 5000
        self.assertGreater(g["dis_frac_gt_4k"], 0.9)

    def test_decision_sampling_covers_the_whole_stream(self):
        from dfp import decisions

        data = text(20000, 4)
        rec = parse_stream(zlib_raw(data, 6), keep_tokens=True)
        seen = []
        real = decisions._search

        def spy(buf, pos, want):
            seen.append(pos)
            return real(buf, pos, want)

        with mock.patch.object(decisions, "MAX_POSITIONS", 50), \
                mock.patch.object(decisions, "_search", spy):
            decisions.extract_decision_features(rec)
        self.assertEqual(len(seen), 50)
        self.assertGreater(max(seen), 0.9 * len(data))


# --- exact re-encoding --------------------------------------------------------------------


class TestReencoding(unittest.TestCase):
    def test_flushed_zlib_stream_is_matched(self):
        from dfp.baselines import zlib_reencode_matches

        data = xml(30000, 7)
        m = zlib_reencode_matches(zlib_raw(data, 6, sync_flush=True), data)
        self.assertTrue(m and all(x.sync_flush for x in m))
        self.assertIn(6, {x.level for x in m})
        self.assertIn("sync flush", m[0].describe())


# --- archive verdict ----------------------------------------------------------------------


def four_parts():
    return [("word/a.xml", xml(30000, 1)), ("word/b.xml", xml(26000, 2)),
            ("word/c.xml", xml(22000, 3)), ("word/d.xml", xml(18000, 4))]


class TestVerdict(unittest.TestCase):
    def test_exact_zlib_proof_overrides_the_model(self):
        path = write("proof.docx", docx(four_parts()[:3]))
        clf = StubClassifier({30000: ("zlib-ng", 0.7), 26000: ("zlib-ng", 0.7),
                              22000: ("zlib", 0.9)}, ["zlib", "zlib-ng", "7zip"])
        v = analyse_archive(str(path), clf)
        self.assertEqual(v.profile, "zlib")
        self.assertNotIn(("mixed-encoders", "inconsistent"), kinds(v))
        self.assertTrue(all(e.proof for e in v.entries))
        self.assertAlmostEqual(v.confidence, 1.0)
        self.assertEqual(v.status, "consistent")

    def test_majority_unknown_gives_an_unknown_verdict(self):
        path = write("unk.docx", docx(four_parts()))
        clf = StubClassifier({22000: ("go-flate", 0.65)}, ["zlib", "go-flate"])
        v = analyse_archive(str(path), clf, reencode=False, verify=False)
        self.assertEqual(v.profile, "unknown")
        self.assertEqual(v.closest, "go-flate")
        self.assertEqual(v.status, "inconclusive")
        self.assertFalse(v.inconsistent)

    def test_weak_disagreement_is_not_reported_as_an_edit(self):
        path = write("weak.docx", docx(four_parts()))
        clf = StubClassifier({30000: ("7zip", 0.65), 26000: ("7zip", 0.65),
                              22000: ("go-flate", 0.65), 18000: ("go-flate", 0.65)},
                             ["zlib", "7zip", "go-flate"])
        v = analyse_archive(str(path), clf, reencode=False, verify=False)
        self.assertIn(("mixed-encoders", "cannot-confirm"), kinds(v))
        self.assertFalse(v.inconsistent)

    def test_unprofiled_producer_is_never_reported_consistent(self):
        path = write("lo.docx", docx(four_parts()[:2], app="Some Unlisted Writer 1.0"))
        clf = StubClassifier({30000: ("zlib", 0.9), 26000: ("zlib", 0.9)}, ["zlib"])
        v = analyse_archive(str(path), clf)
        self.assertEqual(v.profile, "zlib")
        self.assertEqual(v.status, "inconclusive")

    def test_missing_reference_encoders_are_recorded(self):
        path = write("go.docx", docx(four_parts()[:2]))
        clf = StubClassifier({30000: ("go-flate", 0.9), 26000: ("7zip", 0.9)},
                             ["go-flate", "7zip", "zlib"])
        with mock.patch("dfp.encoders.reference_for", return_value=None):
            v = analyse_archive(str(path), clf, reencode=False)
        self.assertEqual(v.verification["reference_encoders_unavailable"], ["7zip", "go-flate"])

    def test_application_profiles_are_not_reported_as_missing_installs(self):
        path = write("app.docx", docx(four_parts()[:2]))
        clf = StubClassifier({30000: ("word", 0.9), 26000: ("go-flate", 0.9)},
                             ["word", "go-flate", "zlib"])
        v = analyse_archive(str(path), clf, reencode=False)
        self.assertIn("word", v.verification["profiles_without_reference_encoder"])
        self.assertNotIn("word", v.verification["reference_encoders_unavailable"])

    # -- exact and structural evidence for mixed encoders -------------------------------------

    @staticmethod
    def _recompress(data, names, how):
        """Recompress the named parts: ``how`` is 'purepy', 'finish' (zlib
        level 6, as a Python script writes) or 'flush' (zlib level 1 with a
        sync flush, ending the way Microsoft Office ends its parts)."""
        from dfp.realfiles import edit_with_python
        from dfp.zipwriter import replace_entry

        for name in names:
            if how == "purepy":
                data = edit_with_python(data, name, "purepy")
                continue
            content = zipfile.ZipFile(io.BytesIO(data)).read(name)
            raw = zlib_raw(content, 1 if how == "flush" else 6, sync_flush=how == "flush")
            data = replace_entry(data, name, content, raw)
        return data

    def _office_like(self, parts=None, app="Microsoft Office Word"):
        """A DOCX whose parts all end like Microsoft Office's, with Word's
        option bits ('super fast') and the given claim."""
        from dfp.realfiles import _claim_with_bits

        parts = parts or four_parts()
        data = self._recompress(docx(parts, app=app), [n for n, _ in parts], "flush")
        return _claim_with_bits(data, app, 3)

    def test_an_edit_that_ends_differently_is_named(self):
        # Word-like parts end with a sync flush; a Python edit ends with zlib's finish
        from dfp.evaluate import edit_outcome

        data = self._recompress(self._office_like(), ["word/d.xml"], "finish")
        path = write("ends.docx", data)
        clf = StubClassifier({30000: ("word", 0.7), 26000: ("word", 0.7),
                              22000: ("word", 0.7), 18000: ("word", 0.7)}, ["word", "zlib"])
        v = analyse_archive(str(path), clf)
        mixed = [f for f in v.findings if f.check == "mixed-encoders"]
        self.assertEqual(mixed[0].kind, "inconsistent")
        self.assertEqual(mixed[0].entries, ["word/d.xml"])
        self.assertTrue(edit_outcome(v, "word/d.xml")["localised"])
        # and the Word claim is contradicted for that part only
        prod = [f for f in v.findings if f.check == "producer" and f.kind == "inconsistent"]
        self.assertEqual(prod[0].entries, ["word/d.xml"])

    def test_an_edit_below_the_proof_size_is_still_caught_by_its_ending(self):
        parts = four_parts()[:3] + [("word/small.xml", xml(1500, 10))]
        data = self._recompress(self._office_like(parts), ["word/small.xml"], "finish")
        path = write("ends-small.docx", data)
        size = zipfile.ZipFile(path).getinfo("word/small.xml").compress_size
        self.assertLess(size, 1024)
        v = analyse_archive(str(path), StubClassifier({}, ["word", "zlib"]))
        mixed = [f for f in v.findings if f.check == "mixed-encoders"]
        self.assertEqual(mixed[0].kind, "inconsistent")
        self.assertEqual(mixed[0].entries, ["word/small.xml"])

    def test_streams_without_matches_are_not_compared(self):
        # an incompressible part (an image) can take another code path in one writer
        parts = four_parts()[:3] + [("word/media/image1.png", random.Random(5).randbytes(3000))]
        data = self._recompress(self._office_like(parts), ["word/media/image1.png"], "finish")
        path = write("image.docx", data)
        v = analyse_archive(str(path), StubClassifier({}, ["word", "zlib"]))
        self.assertFalse(v.inconsistent)
        img = next(e for e in v.entries if e.name == "word/media/image1.png")
        self.assertIsNone(img.proof)  # a stored block is no proof of zlib

    def test_a_tie_names_no_entries(self):
        from dfp.evaluate import edit_outcome

        data = self._recompress(self._office_like(four_parts()[:2]), ["word/b.xml"], "finish")
        path = write("tie.docx", data)
        v = analyse_archive(str(path), StubClassifier({}, ["word", "zlib"]))
        mixed = [f for f in v.findings if f.check == "mixed-encoders"]
        self.assertEqual(mixed[0].kind, "inconsistent")
        self.assertEqual(mixed[0].entries, [])
        self.assertIn("cannot be told", mixed[0].text)
        self.assertFalse(edit_outcome(v, "word/b.xml")["localised"])

    def test_proofs_by_two_encoder_families_name_the_edited_entry(self):
        # same ending (data), but zlib proves three parts and purepy the fourth
        from dfp.evaluate import edit_outcome

        data = self._recompress(docx(four_parts()), ["word/d.xml"], "purepy")
        path = write("two-families.docx", data)
        clf = StubClassifier({30000: ("zlib-ng", 0.7), 26000: ("zlib-ng", 0.7),
                              22000: ("zlib-ng", 0.7), 18015: ("purepy", 0.99)},
                             ["zlib", "zlib-ng", "purepy"])
        v = analyse_archive(str(path), clf)
        mixed = [f for f in v.findings if f.check == "mixed-encoders"]
        self.assertEqual(mixed[0].kind, "inconsistent")
        self.assertEqual(mixed[0].entries, ["word/d.xml"])
        self.assertTrue(edit_outcome(v, "word/d.xml")["localised"])

    def test_small_parts_break_a_tie_between_proven_sides(self):
        # one provable part per side, but a small part only zlib reproduces
        parts = [("word/a.xml", xml(30000, 1)), ("word/d.xml", xml(18000, 4)),
                 ("word/s.xml", xml(1500, 10))]
        data = self._recompress(docx(parts), ["word/d.xml"], "purepy")
        path = write("tie-break.docx", data)
        clf = StubClassifier({30000: ("zlib", 0.7), 18015: ("purepy", 0.99)},
                             ["zlib", "purepy"])
        v = analyse_archive(str(path), clf)
        mixed = [f for f in v.findings if f.check == "mixed-encoders"]
        self.assertEqual(mixed[0].kind, "inconsistent")
        self.assertEqual(mixed[0].entries, ["word/d.xml"])

    def test_confident_disagreement_alone_is_not_an_edit(self):
        # the edited part is confidently labelled 7zip, but nothing exact shows it
        data = self._recompress(docx(four_parts()), ["word/d.xml"], "purepy")
        path = write("confident-only.docx", data)
        clf = StubClassifier({30000: ("zlib", 0.7), 26000: ("zlib", 0.7),
                              22000: ("zlib", 0.7), 18015: ("7zip", 0.99)}, ["zlib", "7zip"])
        with mock.patch("dfp.encoders.reference_for", return_value=None):
            v = analyse_archive(str(path), clf)
        self.assertFalse(v.inconsistent)
        self.assertIn(("mixed-encoders", "cannot-confirm"), kinds(v))

    def test_zlib_family_proofs_do_not_show_two_encoders(self):
        # zlib-ng coincides with zlib on some inputs, so the two proofs are one family
        from dfp.encoders import get_encoder

        enc = get_encoder("zlib-ng")
        if not enc.available():
            raise unittest.SkipTest("zlib-ng binding not installed")
        from dfp.zipwriter import replace_entry

        data = docx(four_parts())
        content = zipfile.ZipFile(io.BytesIO(data)).read("word/d.xml")
        raw = next(r.raw_deflate for st in enc.settings()
                   if (r := enc.compress(content, st)).raw_deflate
                   != zlib_raw(content, int(st.rstrip("f")) if st.rstrip("f").isdigit() else 6))
        path = write("zlib-family.docx", replace_entry(data, "word/d.xml", content, raw))
        clf = StubClassifier({30000: ("zlib", 0.7), 26000: ("zlib", 0.7),
                              22000: ("zlib", 0.7), 18000: ("zlib-ng", 0.99)},
                             ["zlib", "zlib-ng"])
        v = analyse_archive(str(path), clf)
        self.assertNotIn(("mixed-encoders", "inconsistent"), kinds(v))

    def test_large_entry_is_proven_by_an_encoder_proven_elsewhere_in_the_archive(self):
        # a, b and c were written by one encoder (purepy stands in for it); the
        # model mislabels b; d was edited with zlib
        data = self._recompress(docx(four_parts()), ["word/a.xml", "word/b.xml", "word/c.xml"],
                                "purepy")
        path = write("step3a.docx", data)
        clf = StubClassifier({30015: ("purepy", 0.99), 26015: ("word", 0.99),
                              22015: ("purepy", 0.99), 18000: ("zlib", 0.7)},
                             ["purepy", "word", "zlib"])
        v = analyse_archive(str(path), clf)
        b = next(e for e in v.entries if e.name == "word/b.xml")
        self.assertEqual(b.label, "purepy")
        self.assertIn("purepy", b.proof)
        mixed = [f for f in v.findings if f.check == "mixed-encoders"]
        self.assertEqual(mixed[0].kind, "inconsistent")
        self.assertEqual(mixed[0].entries, ["word/d.xml"])

    def test_unknown_entry_is_proven_by_a_leading_reference_encoder(self):
        data = self._recompress(docx(four_parts()), ["word/d.xml"], "purepy")
        path = write("rescue.docx", data)
        # purepy is never a model class; it stands in for a known encoder at
        # a setting the model did not learn
        clf = StubClassifier({30000: ("zlib", 0.9), 26000: ("zlib", 0.9), 22000: ("zlib", 0.9),
                              18015: ("purepy", 0.5)}, ["zlib", "purepy"])
        v = analyse_archive(str(path), clf)
        d = next(e for e in v.entries if e.name == "word/d.xml")
        self.assertEqual(d.label, "purepy")
        self.assertIn("purepy", d.proof)
        self.assertIn(("mixed-encoders", "inconsistent"), kinds(v))

    def test_confident_label_contradicted_by_zlib_is_not_established(self):
        from dfp.producers import check_producer

        # a genuine zlib archive: one small part confidently mislabelled, but
        # zlib reproduces it exactly
        parts = [("word/big.xml", xml(30000, 1)), ("word/big2.xml", xml(26000, 2)),
                 ("word/s.xml", xml(1500, 10))]
        path = write("genuine-confident.docx", docx(parts))
        clf = StubClassifier({30000: ("zlib", 0.7), 26000: ("zlib", 0.7),
                              1500: ("zlib-ng", 0.99)}, ["zlib", "zlib-ng"])
        v = analyse_archive(str(path), clf)
        self.assertFalse(v.inconsistent)
        # a weak verdict never contradicts a claim
        word = {"producer": "Microsoft Office Word", "source": "docProps/app.xml"}
        weak = check_producer(word, "7zip", ["7zip", "word"], [3], verdict_established=False)
        self.assertIn(("producer", "cannot-confirm"), {(f.check, f.kind) for f in weak})
        strong = check_producer(word, "7zip", ["7zip", "word"], [3], verdict_established=True)
        self.assertIn(("producer", "inconsistent"), {(f.check, f.kind) for f in strong})

    # -- producer claims ----------------------------------------------------------------------

    def test_word_parts_that_zlib_reproduces_with_a_flush_are_consistent_with_word(self):
        # some Word builds write exactly what zlib level 1 with a sync flush writes
        path = write("word-zlib1f.docx", self._office_like())
        v = analyse_archive(str(path), StubClassifier({}, ["word", "zlib"]))
        self.assertTrue(any(e.proof for e in v.entries))
        self.assertFalse(v.inconsistent)
        self.assertIn(("producer", "consistent"), kinds(v))

    def test_a_word_claim_on_zlib_finish_parts_is_contradicted(self):
        # python-docx: claims Word, compresses with zlib's finish, 'normal' bits
        path = write("python-docx-like.docx", docx(four_parts(), app="Microsoft Macintosh Word"))
        v = analyse_archive(str(path), StubClassifier({}, ["word", "zlib"]))
        self.assertIn(("producer", "inconsistent"), kinds(v))
        self.assertIn(("option-bits", "inconsistent"), kinds(v))

    def test_a_libreoffice_claim_on_office_structured_parts_is_contradicted(self):
        from dfp.realfiles import _claim_with_bits

        data = _claim_with_bits(self._office_like(), "LibreOffice/24.2.7.2", 0)
        path = write("not-lo.docx", data)
        v = analyse_archive(str(path), StubClassifier({}, ["word", "zlib"]))
        prod = [f for f in v.findings if f.check == "producer"]
        self.assertEqual([f.kind for f in prod], ["inconsistent"])
        self.assertIn("end otherwise", prod[0].text)

    def test_genuine_zlib_producer_is_not_contradicted(self):
        path = write("lo-genuine.docx", docx(four_parts(), app="LibreOffice/24.2.7.2"))
        clf = StubClassifier({30000: ("zlib", 0.7), 26000: ("zlib", 0.7),
                              22000: ("zlib", 0.7), 18000: ("zlib", 0.7)}, ["zlib", "word"])
        v = analyse_archive(str(path), clf)
        self.assertFalse(v.inconsistent)
        self.assertIn(("producer", "consistent"), kinds(v))

    def test_tiny_zlib_parts_never_contradict_a_word_claim(self):
        parts = [(f"word/p{i}.xml", xml(700, i)) for i in range(5)]  # < 1 KiB compressed
        path = write("tiny-word.docx", self._office_like(parts))
        v = analyse_archive(str(path), StubClassifier({}, ["zlib", "word"]))
        self.assertFalse(v.inconsistent)

    def test_pdf_claims_are_not_checked_against_the_zip_producer_table(self):
        # Word's PDF export writes zlib streams; /Creator names the authoring program
        body = zlib.compress(xml(30000, 1))
        pdf = (b"%PDF-1.4\n1 0 obj\n<< /Creator (Microsoft Word) /Length "
               + str(len(body)).encode() + b" /Filter /FlateDecode >>\nstream\n"
               + body + b"\nendstream\nendobj\ntrailer\n<<>>\n%%EOF\n")
        path = write("word-export.pdf", pdf)
        v = analyse_archive(str(path), StubClassifier({}, ["word", "zlib"]))
        self.assertFalse(v.inconsistent)
        self.assertNotIn(("producer", "inconsistent"), kinds(v))


# --- evaluation ---------------------------------------------------------------------------


class TestEvaluation(unittest.TestCase):
    def _corpus(self, sources, app_rows=()):
        from dfp.corpus import Corpus

        rows = [{"source_id": s.id, "content_type": s.content_type, "out_size": s.size,
                 "compressed_size": 1000, "profile": "zlib", "profiles": ["zlib"],
                 "settings": {"zlib": ["6"]}, "labels": ["zlib/6"], "synthetic": False,
                 "origin": "corpus", "split": None, "hash": s.id} for s in sources]
        rows += list(app_rows)
        manifest = {"sources": [{"id": s.id, "content_type": s.content_type, "size": s.size,
                                 "origin": s.origin} for s in sources]}
        return Corpus(np.zeros((len(rows), len(FEATURE_NAMES))), rows, manifest)

    def test_families_join_near_duplicates_and_name_siblings(self):
        from dfp.corpus import directory_sources
        from dfp.evaluate import source_families, split_sources_with_families

        d1, d2, d3 = TMPDIR / "fam1", TMPDIR / "fam2", TMPDIR / "fam3"
        for d in (d1, d2, d3):
            d.mkdir(exist_ok=True)
        base = text(8000, 11)
        (d1 / "crox1h.pcf").write_bytes(base)
        (d1 / "crox1hb.pcf").write_bytes(os.urandom(8000))  # same name stem, other content
        (d2 / "alpha.txt").write_bytes(xml(8000, 12))
        (d3 / "beta.txt").write_bytes(xml(8000, 12)[:7900] + b"y" * 100)  # near duplicate
        (d2 / "gamma.txt").write_bytes(os.urandom(8000))
        srcs = directory_sources([d1, d2, d3])
        by_name = {Path(s.origin).name: s.id for s in srcs}
        corpus = self._corpus(srcs)
        fam, info = source_families(corpus)
        self.assertEqual(fam[by_name["crox1h.pcf"]], fam[by_name["crox1hb.pcf"]])
        self.assertEqual(fam[by_name["alpha.txt"]], fam[by_name["beta.txt"]])
        self.assertNotEqual(fam[by_name["gamma.txt"]], fam[by_name["alpha.txt"]])
        train, test, _ = split_sources_with_families(corpus, 0.4, seed=3)
        for a, b in (("crox1h.pcf", "crox1hb.pcf"), ("alpha.txt", "beta.txt")):
            self.assertEqual(by_name[a] in test, by_name[b] in test)

    def test_same_file_in_two_directories_is_one_source(self):
        from dfp.corpus import directory_sources

        a, b = TMPDIR / "dupa", TMPDIR / "dupb"
        a.mkdir(exist_ok=True)
        b.mkdir(exist_ok=True)
        (a / "lic.txt").write_bytes(text(3000, 5))
        (b / "copy.txt").write_bytes(text(3000, 5))
        seen = set()
        srcs = directory_sources(a, seen=seen) + directory_sources(b, seen=seen)
        self.assertEqual(len(srcs), 1)
        self.assertTrue(srcs[0].id.startswith("file:"))

    def test_app_rows_follow_their_own_split(self):
        from dfp.corpus import Source
        from dfp.evaluate import selection_masks

        srcs = [Source(f"file:{i:016x}", "txt", b"x" * 10, f"/nowhere/{i}") for i in range(2)]
        app = [{"source_id": f"app:word:{i}", "content_type": "docx", "out_size": 10,
                "compressed_size": 900, "profile": "word", "profiles": ["word"],
                "settings": {"word": ["word:saved"]}, "labels": ["word/saved"],
                "synthetic": False, "origin": "app", "split": split, "hash": f"h{i}"}
               for i, split in enumerate(("train", "holdout"))]
        corpus = self._corpus(srcs, app)
        train, test = selection_masks(corpus, {srcs[0].id}, {srcs[1].id})
        self.assertEqual([r["source_id"] for r, m in zip(corpus.rows, train) if m],
                         [srcs[0].id, "app:word:0"])
        self.assertEqual([r["source_id"] for r, m in zip(corpus.rows, test) if m],
                         [srcs[1].id, "app:word:1"])

    def test_end_to_end_metrics_count_unknown_as_a_miss(self):
        from dfp.evaluate import closed_set_metrics

        rows = [{"profiles": ["zlib"], "compressed_size": 2000}] * 4
        preds = [Prediction("zlib", 0.9, False, "", 1, {}, "zlib")] * 2 + [
            Prediction(UNKNOWN, 0.4, True, "", 1, {}, "zlib")] * 2
        m = closed_set_metrics(rows, preds, ["zlib"])
        self.assertEqual(m["accuracy"], 1.0)              # forced choice
        self.assertEqual(m["end_to_end_accuracy"], 0.5)   # what the tool outputs
        self.assertEqual(m["per_profile_end_to_end"][0]["recall"], 0.5)

    def test_end_to_end_metrics_apply_the_minimum_evidence_size(self):
        # below the minimum the analyser says "insufficient evidence": no answer
        from dfp.evaluate import closed_set_metrics, size_band_metrics

        rows = [{"profiles": ["zlib"], "compressed_size": n} for n in (100, 100, 2000, 2000)]
        preds = [Prediction("zlib", 0.9, False, "", 1, {}, "zlib")] * 4
        m = closed_set_metrics(rows, preds, ["zlib"], min_evidence=256)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["end_to_end_accuracy"], 0.5)
        self.assertEqual(m["coverage"], 0.5)
        bands = {b["band"]: b for b in size_band_metrics(rows, preds, min_evidence=256)["bands"]}
        small = next(b for b in bands.values() if b["hi"] <= 256)
        self.assertEqual(small["end_to_end_accuracy"], 0.0)
        self.assertEqual(small["accuracy_on_answered"], 1.0)  # the classifier itself

    def test_real_file_edits_by_the_files_own_encoder_are_not_scored(self):
        # a zlib file recompressed by zlib is exactly what its writer would
        # have produced: no method can detect that, so it is not an edit test
        from dfp.realfiles import evaluate_real_files

        path = write("own-encoder.docx", docx(four_parts()))
        clf = StubClassifier({30000: ("zlib", 0.9), 26000: ("zlib", 0.9), 22000: ("zlib", 0.9),
                              18000: ("zlib", 0.9), 18015: ("zlib", 0.9)}, ["zlib"])
        r = evaluate_real_files(clf, [path], editor="zlib", cross_editor=None)
        self.assertEqual(r["edits"], [])
        self.assertEqual(r["edit_same_encoder_skipped"], 1)
        self.assertIsNone(r["edit_flagged_rate"])

    def test_editor_and_rewrite_choices(self):
        from dfp.realfiles import pick_editor, rewrite_target

        self.assertEqual(pick_editor("7zip"), "zlib")
        ed = pick_editor("zlib")
        self.assertTrue(ed is None or ed != "zlib")
        self.assertTrue(rewrite_target("Microsoft Office Word")[0].startswith("LibreOffice"))
        self.assertEqual(rewrite_target("LibreOffice/24.2"), ("Microsoft Office Word", 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
