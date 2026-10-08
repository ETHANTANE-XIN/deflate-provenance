"""Test suite for DeflateProvenance (stdlib unittest).

Run from the project root:  python -m unittest discover -s tests
Every file the tests write goes to a temporary directory.
"""

import io
import os
import random
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from dfp.bitreader import BitReader, BitStreamError  # noqa: E402
from dfp.containers import detect_container, extract_streams  # noqa: E402
from dfp.covert import (  # noqa: E402
    bytes_to_bits, detect_covert_padding, embed_in_padding, extract_from_padding,
)
from dfp.decisions import DECISION_FEATURE_NAMES, extract_decision_features  # noqa: E402
from dfp.deflate import DeflateError, inflate, parse_stream  # noqa: E402
from dfp.encoders import get_encoder, list_encoders  # noqa: E402
from dfp.features import FEATURE_NAMES, features_to_vector, stream_features  # noqa: E402
from dfp.huffman import code_cost, huffman_lengths, kraft_ok, limited_lengths  # noqa: E402
from dfp.signatures import evaluate_signatures  # noqa: E402

TMP = tempfile.TemporaryDirectory()
TMPDIR = Path(TMP.name)
os.environ.setdefault("DFP_CACHE", str(TMPDIR / "cache"))
# never read or write the user's own model store while testing
os.environ["DFP_MODELS"] = str(TMPDIR / "store")


def zc(data, level=6, strategy=zlib.Z_DEFAULT_STRATEGY, mem_level=8):
    c = zlib.compressobj(level, zlib.DEFLATED, -15, mem_level, strategy)
    return c.compress(data) + c.flush()


def text(n, seed=1):
    from dfp.corpus import _text

    return _text(random.Random(seed), n)


def xml(n, seed=1):
    from dfp.corpus import _xml

    return _xml(random.Random(seed), n)


def make_docx(entries, app=None, level=6):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as z:
        if app is not None:
            z.writestr("docProps/app.xml",
                       f"<Properties><Application>{app}</Application></Properties>")
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


def real_profiles_available():
    return [e for e in list_encoders(only_available=True, include_synthetic=False)
            if e.reference]


# --- parsing ----------------------------------------------------------------------


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
        rnd = random.Random(42)
        samples = [b"", b"A", b"the quick brown fox " * 300, b"\x00" * 40000,
                   os.urandom(20000), bytes(rnd.randrange(256) for _ in range(15000)),
                   xml(30000)]
        strategies = [zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED, zlib.Z_HUFFMAN_ONLY,
                      zlib.Z_RLE, zlib.Z_FIXED]
        checked = 0
        for data in samples:
            for level in range(10):
                for strat in strategies:
                    rec = inflate(zc(data, level, strat), strict=True)
                    self.assertEqual(rec.output, data)
                    checked += 1
        self.assertEqual(checked, 350)

    def test_literal_histogram_counts_every_literal(self):
        data = text(20000)
        rec = inflate(zc(data), strict=True)
        coded = sum(sum(b.literal_hist) for b in rec.blocks if b.btype != 0)
        self.assertEqual(coded, rec.n_literals)

    def test_consumed_size_matches(self):
        blob = zc(b"forensics " * 500)
        self.assertEqual(inflate(blob, strict=True).compressed_bytes, len(blob))

    def test_stored_block_level0(self):
        rec = inflate(zc(b"X" * 200000, level=0), strict=True)
        self.assertEqual(rec.block_type_counts()["dynamic"], 0)
        self.assertGreater(rec.block_type_counts()["stored"], 0)

    def test_truncated_stream_reports_error(self):
        blob = zc(text(5000))
        for cut in (1, 5, len(blob) // 2, len(blob) - 1):
            self.assertIsNotNone(parse_stream(blob[:cut], strict=False).error)

    def test_reserved_block_type_raises(self):
        with self.assertRaises(DeflateError):
            inflate(bytes([0b00000111, 0, 0, 0]), strict=True)


# --- optimal Huffman ------------------------------------------------------------------


class TestHuffman(unittest.TestCase):
    def test_limited_lengths_are_valid_and_optimal(self):
        rnd = random.Random(3)
        for _ in range(300):
            n = rnd.randint(2, 30)
            freqs = [rnd.choice([0, 1, 2, 5, 100, rnd.randint(1, 10**5)]) for _ in range(n)]
            if sum(1 for f in freqs if f) < 2:
                continue
            got = limited_lengths(freqs, 15)
            self.assertTrue(kraft_ok(got, 15))
            # without a binding limit the result equals unrestricted Huffman cost
            self.assertEqual(code_cost(freqs, got), code_cost(freqs, huffman_lengths(freqs)))

    def test_length_limit_is_enforced(self):
        fib = [1, 1]
        for _ in range(30):
            fib.append(fib[-1] + fib[-2])
        got = limited_lengths(fib, 15)
        self.assertEqual(max(got), 15)
        self.assertTrue(kraft_ok(got, 15))


# --- features --------------------------------------------------------------------------


class TestFeatures(unittest.TestCase):
    def test_vector_length_stable(self):
        _, feats = stream_features(zc(b"feature vector test " * 300))
        self.assertEqual(len(features_to_vector(feats)), len(FEATURE_NAMES))
        self.assertTrue(set(DECISION_FEATURE_NAMES) <= set(FEATURE_NAMES))

    def test_features_finite(self):
        import math

        _, feats = stream_features(zc(os.urandom(8000)))
        for name, v in feats.items():
            self.assertTrue(math.isfinite(v), f"{name} not finite: {v}")

    def test_decision_features_separate_greedy_from_lazy(self):
        data = text(60000)
        fast = extract_decision_features(parse_stream(zc(data, 1), keep_tokens=True))
        slow = extract_decision_features(parse_stream(zc(data, 9), keep_tokens=True))
        # level 1 (greedy, short chains) rarely takes the longest match; level 9 does
        self.assertLess(fast["dec_match_longest_frac"], 0.6)
        self.assertGreater(slow["dec_match_longest_frac"], 0.95)
        # zlib's lazy evaluation shows as literals followed by a longer match
        self.assertGreater(slow["dec_lazy_defer_frac"], 0.5)

    def test_zlib_trees_are_optimal(self):
        f = extract_decision_features(parse_stream(zc(text(40000)), keep_tokens=True))
        self.assertAlmostEqual(f["huf_lit_excess_mean"], 0.0, places=6)

    def test_decision_features_need_tokens(self):
        f = extract_decision_features(parse_stream(zc(text(5000)), keep_tokens=False))
        self.assertEqual(f["dec_match_longest_frac"], 0.0)


# --- encoders ----------------------------------------------------------------------------


class TestEncoders(unittest.TestCase):
    def test_all_available_encoders_roundtrip(self):
        data = xml(12000)
        for enc in list_encoders(only_available=True):
            self.assertTrue(enc.version())
            for setting, raw in enc.compress_many(data, enc.settings()).items():
                self.assertEqual(inflate(raw, strict=True).output, data, f"{enc.name}/{setting}")

    def test_zlib_levels_are_one_to_nine_plus_flushed_variants(self):
        settings = get_encoder("zlib").settings()
        self.assertEqual(settings[:9], [str(i) for i in range(1, 10)])
        self.assertEqual(settings[9:], ["1f", "6f", "9f"])
        raw = get_encoder("zlib").compress(xml(5000), "6f").raw_deflate
        self.assertTrue(raw.endswith(b"\x00\x00\xff\xff\x03\x00"))
        self.assertEqual(zlib.decompress(raw, -15), xml(5000))

    def test_purepy_is_synthetic(self):
        enc = get_encoder("purepy")
        self.assertTrue(enc.synthetic)
        self.assertNotIn(enc, list_encoders(include_synthetic=False))


# --- corpus ------------------------------------------------------------------------------


class TestCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dfp.corpus import build_corpus, synthetic_sources

        names = ["zlib", "java"] + [e.name for e in real_profiles_available()
                                    if e.name != "zlib"][:2]
        cls.sources = synthetic_sources(content_types=["text", "xml"], sizes=[3000, 20000],
                                        per_combo=1)
        cls.corpus = build_corpus(cls.sources, encoders=names, workers=1)

    def test_sources_are_unique(self):
        datas = [s.data for s in self.sources]
        self.assertEqual(len(set(datas)), len(datas))

    def test_identical_outputs_share_one_row_with_a_label_set(self):
        hashes = [r["hash"] for r in self.corpus.rows]
        self.assertEqual(len(hashes), len(set((r["source_id"], r["hash"])
                                              for r in self.corpus.rows)))
        multi = [r for r in self.corpus.rows if len(r["labels"]) > 1]
        self.assertTrue(multi)

    def test_java_shares_the_zlib_profile_when_verified(self):
        sharing = self.corpus.manifest["profile_sharing"]
        if "java" not in sharing:
            self.skipTest("Java is not installed")
        self.assertEqual(sharing["java"]["profile"], "zlib")
        self.assertGreaterEqual(sharing["java"]["identical_rate"], 0.95)

    def test_manifest_records_versions(self):
        for enc in self.corpus.manifest["encoders"]:
            self.assertTrue(enc["version"])

    def test_save_load_and_regenerate(self):
        from dfp.corpus import Corpus, regenerate_streams

        d = TMPDIR / "corpus"
        self.corpus.save(d)
        c2 = Corpus.load(d)
        self.assertTrue(np.allclose(c2.X, self.corpus.X))
        sid = self.corpus.rows[0]["source_id"]
        raws = regenerate_streams(c2, {sid})
        self.assertTrue(raws)

    def test_split_by_source_has_no_overlap(self):
        from dfp.evaluate import split_sources

        train, test = split_sources(self.corpus, 0.5, seed=1)
        self.assertFalse(train & test)
        self.assertTrue(train and test)


# --- classifier -----------------------------------------------------------------------------


class TestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dfp.corpus import build_corpus, synthetic_sources
        from dfp.training import train_model

        profs = real_profiles_available()
        if len(profs) < 2:
            raise unittest.SkipTest("needs two real encoder profiles")
        names = [e.name for e in profs[:3]]
        src = synthetic_sources(content_types=["text", "xml", "json"],
                                sizes=[2000, 8000, 24000], per_combo=2)
        cls.corpus = build_corpus(src, encoders=names, workers=2)
        cls.clf = train_model(cls.corpus, n_estimators=30)

    def test_calibration_uses_held_out_sources(self):
        cal = self.clf.calibration
        self.assertGreater(cal["held_out_sources"], 0)
        self.assertIn("min_confidence", cal)
        self.assertGreater(self.clf.max_distance, 0)
        # the high-precision threshold is never below the ordinary one
        self.assertGreaterEqual(self.clf.strong_confidence, self.clf.min_confidence)
        # the minimum evidence size never drops below the floor, and the
        # held-out measurement is recorded beside it
        self.assertGreaterEqual(self.clf.min_evidence_bytes, self.clf.min_evidence_floor)
        self.assertIn("min_evidence_bytes_measured", cal)
        # the novelty rule reports what it does on known and simulated unseen encoders
        nov = cal["novelty"]
        self.assertIn("rejected_by_novelty", nov["known"])
        self.assertTrue(nov["unseen_rejected_by_novelty"])

    def test_prediction_has_setting_and_explanation(self):
        row = next(i for i, r in enumerate(self.corpus.rows) if r["profile"])
        p = self.clf.predict_one(self.corpus.X[row], explain=5)
        self.assertIn(p.raw_top, self.clf.classes)
        self.assertTrue(p.explanation)
        if not p.abstained:
            self.assertIsNotNone(p.setting)

    def test_explanations_add_up_to_the_forest_probability(self):
        X = self.clf._scale(self.corpus.X[:20])
        proba = self.clf._forest.predict_proba(X)
        top = proba.argmax(axis=1)
        bias, contrib = self.clf._forest.contributions(X, top)
        self.assertTrue(np.allclose(bias + contrib.sum(axis=1), proba[np.arange(20), top]))

    def test_save_load(self):
        from dfp.ml import ProvenanceClassifier

        path = TMPDIR / "m.json"
        self.clf.save(str(path))
        clf2 = ProvenanceClassifier.load(str(path))
        a = self.clf.predict_one(self.corpus.X[0])
        b = clf2.predict_one(self.corpus.X[0])
        self.assertEqual(a.raw_top, b.raw_top)
        self.assertAlmostEqual(a.confidence, b.confidence, places=4)

    def test_word_claim_on_zlib_archive_is_flagged(self):
        from dfp.aggregate import analyse_archive

        if "zlib" not in self.clf.classes:
            self.skipTest("zlib profile not in the model")
        path = TMPDIR / "claims-word.docx"
        path.write_bytes(make_docx([("word/document.xml", xml(40000, 5)),
                                    ("word/styles.xml", xml(20000, 6))],
                                   app="Microsoft Office Word"))
        v = analyse_archive(str(path), self.clf)
        kinds = {(f.check, f.kind) for f in v.findings}
        self.assertIn(("option-bits", "inconsistent"), kinds)
        self.assertIn(("producer", "inconsistent"), kinds)

    def test_stored_entry_is_insufficient_evidence(self):
        from dfp.aggregate import INSUFFICIENT, analyse_archive

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(zipfile.ZipInfo("stored.bin"), b"x" * 5000)
            z.writestr("deflated.xml", xml(20000), compress_type=zipfile.ZIP_DEFLATED)
        path = TMPDIR / "mixed-methods.zip"
        path.write_bytes(buf.getvalue())
        v = analyse_archive(str(path), self.clf, reencode=False)
        stored = next(e for e in v.entries if e.name == "stored.bin")
        self.assertEqual(stored.status, INSUFFICIENT)


class TestModelStore(unittest.TestCase):
    def test_bundled_default_model_loads_and_flags_a_false_word_claim(self):
        from dfp import modelstore
        from dfp.aggregate import analyse_archive

        path = modelstore.resolve("default")
        self.assertTrue(str(path).startswith(str(modelstore.BUNDLED_DIR)))
        clf = modelstore.load()  # fails if the features changed without retraining
        self.assertIn("zlib", clf.classes)
        self.assertGreater(clf.training.get("training_sources", 0), 100)
        doc = TMPDIR / "store-claims-word.docx"
        doc.write_bytes(make_docx([("word/document.xml", xml(30000, 8))],
                                  app="Microsoft Office Word"))
        v = analyse_archive(str(doc), clf)
        self.assertEqual(v.profile, "zlib")
        self.assertIn(("producer", "inconsistent"), {(f.check, f.kind) for f in v.findings})

    def test_local_store_overrides_bundled_and_gz_roundtrip(self):
        from dfp import modelstore
        from dfp.ml import ProvenanceClassifier

        bundled = modelstore.load("default")
        path = modelstore.save(bundled, "default")
        self.assertTrue(path.name.endswith(".json.gz"))
        self.assertEqual(modelstore.resolve("default"), path)
        again = ProvenanceClassifier.load(str(path))
        self.assertEqual(again.classes, bundled.classes)
        names = {(m["name"], m["where"]): m["active"] for m in modelstore.list_models()}
        self.assertTrue(names[("default", "local")])
        self.assertFalse(names[("default", "bundled")])
        path.unlink()
        self.assertEqual(modelstore.resolve("default").parent, modelstore.BUNDLED_DIR)

    def test_missing_model_names_what_exists(self):
        from dfp import modelstore

        with self.assertRaises(FileNotFoundError) as ctx:
            modelstore.resolve("no-such-model")
        self.assertIn("default", str(ctx.exception))


class TestVerification(unittest.TestCase):
    def test_reproduced_by_reference_encoder(self):
        from dfp.aggregate import reproduced_by

        data = xml(20000, 4)
        self.assertTrue(reproduced_by("zlib", zc(data, 7), data))
        purepy = get_encoder("purepy").compress(data, "lazy").raw_deflate
        self.assertFalse(reproduced_by("zlib", purepy, data))
        if get_encoder("libdeflate").available():
            raw = get_encoder("libdeflate").compress(data, "6").raw_deflate
            self.assertTrue(reproduced_by("libdeflate", raw, data))
            self.assertFalse(reproduced_by("zlib", raw, data))


class TestVote(unittest.TestCase):
    def test_vote_is_weighted_by_compressed_size(self):
        from dfp.aggregate import ATTRIBUTED, EntryAnalysis, aggregate_votes
        from dfp.ml import Prediction

        def entry(label, size):
            p = Prediction(label, 0.9, False, "attributed", 1.0, {label: 0.9}, label)
            return EntryAnalysis(label, "zip", "deflate", ATTRIBUTED, compressed_size=size,
                                 prediction=p)

        dist, top, share = aggregate_votes([entry("a", 100), entry("a", 100), entry("b", 1000)])
        self.assertEqual(top, "b")
        self.assertAlmostEqual(share, 1000 / 1200)


# --- containers and claims --------------------------------------------------------------------


class TestContainers(unittest.TestCase):
    def test_detect_gzip(self):
        import gzip

        self.assertEqual(detect_container(gzip.compress(b"hello"), "x.gz"), "gzip")

    def test_gzip_extraction(self):
        import gzip

        path = TMPDIR / "t.gz"
        path.write_bytes(gzip.compress(b"content " * 500))
        rep = extract_streams(str(path))
        self.assertEqual(rep.container, "gzip")
        self.assertEqual(zlib.decompress(rep.streams[0].payload, -15), b"content " * 500)

    def test_zip_extraction_and_entries(self):
        path = TMPDIR / "t.zip"
        path.write_bytes(make_docx([("a.txt", b"alpha " * 400), ("b.txt", b"beta " * 400)]))
        rep = extract_streams(str(path))
        self.assertEqual(len(rep.streams), 2)
        self.assertEqual(len(rep.entries), 2)
        for s in rep.streams:
            self.assertEqual(zlib.crc32(zlib.decompress(s.payload, -15)) & 0xFFFFFFFF,
                             s.declared_crc32)

    def test_ooxml_claim_harvested(self):
        path = TMPDIR / "c.docx"
        path.write_bytes(make_docx([("word/document.xml", b"<w/>" * 100)],
                                   app="LibreOffice/24.2"))
        rep = extract_streams(str(path))
        self.assertEqual(rep.claims["producer"], "LibreOffice/24.2")
        self.assertEqual(rep.claims["source"], "docProps/app.xml")
        self.assertEqual(rep.metadata["option_bits_seen"], [0])

    def test_jar_manifest_claim(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\nCreated-By: 21.0.10 (Ubuntu)\n")
        path = TMPDIR / "t.jar"
        path.write_bytes(buf.getvalue())
        self.assertEqual(extract_streams(str(path)).claims["producer"], "21.0.10 (Ubuntu)")

    def test_pdf_flatedecode_extraction(self):
        body_a = b"BT /F1 12 Tf (forensic evidence page one) Tj ET " * 60
        body_b = b"BT /F1 12 Tf (page two content here) Tj ET " * 60
        s1, s2 = zlib.compress(body_a, 6), zlib.compress(body_b, 6)
        pdf = bytearray(b"%PDF-1.7\n")
        pdf += b"1 0 obj\n<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(s1)
        pdf += s1 + b"\nendstream\nendobj\n"
        pdf += b"2 0 obj\n<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(s2)
        pdf += s2 + b"\nendstream\nendobj\n%%EOF\n"
        path = TMPDIR / "t.pdf"
        path.write_bytes(bytes(pdf))
        rep = extract_streams(str(path))
        outs = [parse_stream(s.payload, strict=True).output for s in rep.streams]
        self.assertEqual(outs, [body_a, body_b])


class TestProducers(unittest.TestCase):
    def test_patterns(self):
        from dfp.producers import match_producer

        self.assertEqual(match_producer("Microsoft Office Word").key, "microsoft-word")
        self.assertEqual(match_producer("Microsoft Macintosh Word").key, "microsoft-word")
        self.assertEqual(match_producer("LibreOffice/24.2.7.2$Linux_X86_64").key, "libreoffice")
        self.assertEqual(
            match_producer("Microsoft Excel Compatible / Openpyxl 3.1.5").key, "openpyxl")
        self.assertIsNone(match_producer("Some Unknown Writer"))

    def test_check_logic(self):
        from dfp.producers import check_producer

        word = {"producer": "Microsoft Office Word", "source": "docProps/app.xml"}
        f = check_producer(word, "go-flate", ["zlib", "go-flate", "word"], [0])
        self.assertEqual({(x.check, x.kind) for x in f},
                         {("producer", "inconsistent"), ("option-bits", "inconsistent")})
        # some Word builds compress exactly as zlib does: no contradiction
        f = check_producer(word, "zlib", ["zlib", "word"], [3])
        self.assertTrue(all(x.kind == "consistent" for x in f))
        lo = {"producer": "LibreOffice/24.2", "source": "meta.xml"}
        f = check_producer(lo, "zlib", ["zlib"], [0])
        self.assertTrue(all(x.kind == "consistent" for x in f))
        f = check_producer(lo, "go-flate", ["zlib", "go-flate"], [0])
        self.assertIn(("producer", "inconsistent"), {(x.check, x.kind) for x in f})
        f = check_producer(lo, "unknown", ["zlib"], [0])
        self.assertIn(("producer", "cannot-confirm"), {(x.check, x.kind) for x in f})


# --- ZIP rewriting, baselines, adversarial ---------------------------------------------------


class TestZipWriter(unittest.TestCase):
    def setUp(self):
        self.data = make_docx([("word/document.xml", xml(8000)), ("a.txt", text(3000))],
                              app="LibreOffice/24.2")

    def test_roundtrip_keeps_streams(self):
        from dfp.zipwriter import read_zip, write_zip

        entries, comment = read_zip(self.data)
        again = write_zip(entries, comment)
        z = zipfile.ZipFile(io.BytesIO(again))
        self.assertIsNone(z.testzip())
        self.assertEqual([e.raw for e in read_zip(again)[0]], [e.raw for e in entries])

    def test_impersonate_rewrites_claim_without_touching_streams(self):
        from dfp.zipwriter import read_zip
        from dfp.realfiles import _word_claim

        forged = _word_claim(self.data)
        rep = extract_streams("f.docx", data=forged)
        self.assertEqual(rep.claims["producer"], "Microsoft Office Word")
        self.assertEqual(rep.metadata["option_bits_seen"], [3])
        before = {e.name: e.raw for e in read_zip(self.data)[0] if e.name != "docProps/app.xml"}
        after = {e.name: e.raw for e in read_zip(forged)[0] if e.name != "docProps/app.xml"}
        self.assertEqual(before, after)

    def test_edit_replaces_one_entry(self):
        from dfp.realfiles import edit_with_python
        from dfp.zipwriter import read_zip

        edited = edit_with_python(self.data, "word/document.xml", "zlib")
        z = zipfile.ZipFile(io.BytesIO(edited))
        self.assertIsNone(z.testzip())
        self.assertTrue(z.read("word/document.xml").endswith(b"<!-- edited -->"))
        before = {e.name: e.raw for e in read_zip(self.data)[0]}
        after = {e.name: e.raw for e in read_zip(edited)[0]}
        self.assertNotEqual(before["word/document.xml"], after["word/document.xml"])
        self.assertEqual(before["a.txt"], after["a.txt"])


class TestBaselines(unittest.TestCase):
    def test_brute_force_zlib_returns_every_match(self):
        from dfp.baselines import zlib_reencode_matches

        matches = zlib_reencode_matches(zc(text(20000), 6))
        self.assertIn((6, 8), [(m.level, m.mem_level) for m in matches])
        self.assertFalse(any(m.sync_flush for m in matches))
        self.assertEqual(zlib_reencode_matches(get_encoder("purepy").compress(
            text(5000), "lazy").raw_deflate), [])

    def test_metadata_features_shape(self):
        from dfp.baselines import METADATA_FEATURES, metadata_features

        rep = extract_streams("x.docx", data=make_docx([("a.xml", xml(3000))]))
        self.assertEqual(len(metadata_features(rep)), len(METADATA_FEATURES))

    def test_preflate_estimate(self):
        from dfp.baselines import preflate_binary, preflate_estimate

        if not preflate_binary():
            self.skipTest("tools/preflate-estimate not built")
        res = preflate_estimate([zc(xml(30000, 9), 6)])
        self.assertTrue(res[0]["ok"])
        self.assertIn("hash_algorithm", res[0])
        self.assertIn(res[0]["label"], {"zlib", "zlib-ng", "libdeflate", "chromium-zlib",
                                        "miniz", "unknown"})


class TestAdversarial(unittest.TestCase):
    def test_metadata_normalisation_preserves_bitstream(self):
        from dfp.adversarial import metadata_robustness

        path = TMPDIR / "adv.zip"
        path.write_bytes(make_docx([("doc.xml", b"<x>" + b"data " * 500 + b"</x>")]))
        r = metadata_robustness(str(path))
        self.assertTrue(r.metadata_changed)
        self.assertFalse(r.bitstream_changed)

    def test_reencode_panel_lists_all_matches(self):
        from dfp.adversarial import reencode_panel

        r = reencode_panel(zc(b"reproduce me exactly " * 300, level=6))
        self.assertTrue(r.matched)
        self.assertTrue(any(m.startswith("zlib/level 6") for m in r.matches))


class TestSignatures(unittest.TestCase):
    def test_huffman_only_rule(self):
        rec = parse_stream(zc(b"xy " * 500, strategy=zlib.Z_HUFFMAN_ONLY), strict=True)
        self.assertIn("no-lz77-matches", {h.rule for h in evaluate_signatures(rec)})

    def test_rle_rule(self):
        rec = parse_stream(zc(b"A" * 5000, strategy=zlib.Z_RLE), strict=True)
        self.assertIn("distance-1-only", {h.rule for h in evaluate_signatures(rec)})


class TestCovertExtension(unittest.TestCase):
    def test_padding_roundtrip_output_unchanged(self):
        data = b"secret carrier test " * 137
        raw = zc(data)
        msg_bits = bytes_to_bits(b"K")
        stego = embed_in_padding(raw, msg_bits)
        self.assertEqual(zlib.decompress(stego, -15), data)
        rec = parse_stream(stego, strict=False)
        if rec.final_pad_bits:
            self.assertEqual(extract_from_padding(stego, rec.final_pad_bits),
                             msg_bits[: rec.final_pad_bits])

    def test_detector_flags_nonzero_padding(self):
        raw = zc(b"detect me " * 200)
        rec = parse_stream(raw, strict=False)
        if rec.final_pad_bits:
            self.assertTrue(detect_covert_padding(embed_in_padding(raw, [1] * rec.final_pad_bits))
                            .suspicious)


class TestMinimumEvidence(unittest.TestCase):
    def test_dip_in_large_band_does_not_raise_the_minimum(self):
        from dfp.ml import ProvenanceClassifier

        clf = ProvenanceClassifier()
        sizes, correct = [], []
        # small streams unreliable, mid sizes reliable, largest band dips to 85%
        for size, acc in [(100, 0.7), (500, 0.97), (2000, 0.95), (8000, 1.0),
                          (30000, 0.99), (90000, 0.85)]:
            n = 40
            k = int(round(acc * n))
            sizes += [size] * n
            correct += [True] * k + [False] * (n - k)
        got = clf._reliable_size(np.array(sizes), np.array(correct))
        self.assertEqual(got, 256)


class TestEvaluationHelpers(unittest.TestCase):
    def test_size_bands_and_reliability(self):
        from dfp.evaluate import size_band_metrics
        from dfp.ml import Prediction

        rows, preds = [], []
        for size, ok in [(100, False)] * 30 + [(2000, True)] * 30 + [(20000, True)] * 30:
            rows.append({"compressed_size": size, "profiles": ["zlib"]})
            lab = "zlib" if ok else "go-flate"
            preds.append(Prediction(lab, 0.9, False, "attributed", 1.0, {lab: 0.9}, lab))
        sb = size_band_metrics(rows, preds)
        self.assertEqual(sb["smallest_reliable_compressed_size"], 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
