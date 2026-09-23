"""Offline integration checks for six-view input, source preservation and output safety."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phone_normalizer.api import NormalizerConfig
from phone_normalizer.uv import FACES, normalize_face, process_six_views
from phone_normalizer.uv_cli import main
from phone_normalizer.uv_profile import load_profile


class FakeBackend:
    name = "birefnet"
    model_name = "ZhengPeng7/BiRefNet_dynamic"
    device = "cpu"

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at
        self.closed = False

    def segment(self, image):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("test inference failure")
        rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        rgba[:, :, :3] = 255  # A backend must not replace the source photograph.
        rgba[3:-3, 2:-2, 3] = 255
        rgba[7, 7, 3] = 16  # Connected internal soft alpha must survive.
        return Image.fromarray(rgba)

    def close(self):
        self.closed = True


def settings(**kwargs):
    return NormalizerConfig(crop_threshold=128, remove_support=False, deskew=False, **kwargs)


def make_sources(root):
    source = root / "source"
    source.mkdir()
    for face in FACES:
        Image.new("RGB", (60, 40), (12, 36, 70)).save(source / f"{face.title()}.png")
    return source


class UVPipelineTests(unittest.TestCase):
    def test_source_rgb_soft_alpha_and_orientation(self):
        source = Image.new("RGB", (60, 40), (12, 36, 70))
        saved = np.asarray(source).copy()
        for face, size in (("front", (56, 34)), ("left", (34, 56)), ("right", (34, 56))):
            output, info = normalize_face(source, face, backend=FakeBackend(), config=settings())
            self.assertEqual(output.size, size)
            rgba = np.asarray(output)
            self.assertTrue(np.all(rgba[:, :, :3][rgba[:, :, 3] > 0] == [12, 36, 70]))
            self.assertEqual(int(np.count_nonzero(rgba[:, :, 3] == 16)), 1)
            self.assertEqual(info["uv_fit_validation"], "not_performed")
        np.testing.assert_array_equal(np.asarray(source), saved)

    def test_batch_reuses_backend_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_sources(root)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            backend = FakeBackend()
            output = root / "output"
            report = process_six_views(source, output, config=settings(), backend_instance=backend)
            self.assertTrue(report["success"])
            self.assertEqual(backend.calls, 6)
            self.assertFalse(backend.closed)  # Injected backend belongs to caller.
            self.assertEqual(len(list(output.glob("*.png"))), 6)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            snapshot = (output / "front.png").read_bytes()
            with self.assertRaises(FileExistsError):
                process_six_views(source, output, config=settings(), backend_instance=backend)
            self.assertEqual(backend.calls, 6)
            self.assertEqual(snapshot, (output / "front.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "separate output"):
                process_six_views(source, source, backend_instance=backend)

    def test_validate_all_inputs_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_sources(root)
            backend = FakeBackend()
            (source / "Bottom.png").unlink()
            with self.assertRaisesRegex(ValueError, "bottom"):
                process_six_views(source, root / "out", backend_instance=backend)
            Image.new("RGB", (60, 40)).save(source / "Bottom.png")
            Image.new("RGB", (60, 40)).save(source / "FRONT.jpg")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                process_six_views(source, root / "out", backend_instance=backend)
            (source / "FRONT.jpg").unlink()
            Image.new("RGB", (200, 200)).save(source / "Bottom.png")
            with self.assertRaises(ValueError):
                process_six_views(source, root / "out", backend_instance=backend,
                                  config=settings(max_geometry_pixels=10_000))
            self.assertEqual(backend.calls, 0)
            self.assertFalse((root / "out").exists())

    def test_failed_batch_keeps_successes_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_sources(root)
            with self.assertRaisesRegex(RuntimeError, "test inference"):
                process_six_views(source, root / "out", config=settings(),
                                  backend_instance=FakeBackend(fail_at=2))
            report = json.loads((root / "out/manifest.json").read_text())
            self.assertFalse(report["success"])
            self.assertEqual(list(report["faces"]), ["front"])
            self.assertIn("test inference failure", report["error"])
            self.assertTrue((root / "out/front.png").is_file())
            self.assertFalse((root / "out/back.png").exists())

    def test_bad_profile_settings_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_sources(root)
            backend = FakeBackend()
            with self.assertRaisesRegex(ValueError, "crop_threshold"):
                process_six_views(source, root / "out", backend_instance=backend,
                                  uv_profile="iphone12pro-user-crop-v1", config=NormalizerConfig())
            self.assertEqual(backend.calls, 0)
            with self.assertRaisesRegex(ValueError, "crop_threshold"):
                normalize_face(Image.new("RGB", (60, 40)), "front", backend=backend,
                               config=NormalizerConfig(), profile=load_profile("iphone12pro-user-crop-v1"))
            self.assertEqual(backend.calls, 0)

    def test_cli_fails_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main([str(Path(tmp) / "absent"), "--output-dir", str(Path(tmp) / "out")]), 1)


if __name__ == "__main__":
    unittest.main()
