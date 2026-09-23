#!/usr/bin/env python3
"""Offline regressions for precise crop geometry; no model weights required."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.geometry_postprocess import (  # noqa: E402
    DEFAULT_MAX_GEOMETRY_PIXELS,
    rotate_alpha_safe,
)
from scripts.precise_geometry import normalize_candidate  # noqa: E402


def synthetic_phone() -> np.ndarray:
    image = Image.new("RGBA", (640, 1000), (0, 0, 0, 0))
    ImageDraw.Draw(image).rounded_rectangle(
        (100, 100, 499, 899), radius=40, fill=(100, 100, 100, 255)
    )
    return np.asarray(image).copy()


class PreciseGeometryTests(unittest.TestCase):
    def test_subpixel_deskew_keeps_known_subject_dimensions(self) -> None:
        phone = synthetic_phone()
        # A 0.03-degree roll previously produced 402x802 instead of 400x800.
        for angle in (0.0, 0.03, 0.1, 0.4, 1.0, 5.0, 10.0):
            with self.subTest(angle=angle):
                source = rotate_alpha_safe(phone, angle)
                original = source.copy()
                output, info = normalize_candidate(source, remove_support=False)
                self.assertEqual(output.shape, (800, 400, 4))
                self.assertEqual(info["output_size"], [400, 800])
                self.assertTrue(np.array_equal(source, original))
                self.assertFalse(np.shares_memory(output, source))
                core = output[..., 3] >= 128
                self.assertTrue(core[0, :].any() and core[-1, :].any())
                self.assertTrue(core[:, 0].any() and core[:, -1].any())
                # Rounded corners must stay transparent; touching the four
                # sides does not mean replacing the real silhouette by a box.
                self.assertEqual(int(output[0, 0, 3]), 0)
                self.assertEqual(info["algorithm_version"], "precise-geometry-v1")
                self.assertNotIn("experiment", info)
                json.dumps(info, allow_nan=False)

    def test_preserves_soft_alpha_and_original_rgb_inside_crop(self) -> None:
        source = synthetic_phone()
        source[300, 200] = (41, 93, 167, 16)
        source[301, 200] = (52, 81, 119, 127)
        original = source.copy()
        output, info = normalize_candidate(
            source, remove_support=False, deskew=False
        )
        self.assertEqual(info["crop_box"], [100, 100, 500, 900])
        self.assertTrue(np.array_equal(output[200, 100], source[300, 200]))
        self.assertTrue(np.array_equal(output[201, 100], source[301, 200]))
        self.assertTrue(np.array_equal(source, original))
        self.assertEqual(info["soft_pixels_preserved_inside_crop"], 2)

    def test_discards_isolated_speck_but_retains_connected_soft_region(self) -> None:
        source = synthetic_phone()
        source[20, 20] = (100, 100, 100, 255)
        # A low-alpha patch in the subject remains connected at alpha > 0.
        source[300:310, 200:210, 3] = 8
        output, info = normalize_candidate(
            source, remove_support=False, deskew=False
        )
        self.assertEqual(info["cleanup_before_rotation"]["removed_pixels"], 1)
        self.assertEqual(info["cleanup_before_rotation"]["removed_alpha_sum"], 255)
        self.assertTrue(np.all(output[200:210, 100:110, 3] == 8))
        self.assertEqual(info["output_size"], [400, 800])

    def test_empty_or_insufficient_alpha_fails_explicitly(self) -> None:
        source = np.zeros((30, 40, 4), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "No subject reaches alpha"):
            normalize_candidate(source, remove_support=False, deskew=False)
        source[5:25, 5:35] = (100, 100, 100, 64)
        with self.assertRaisesRegex(ValueError, "No subject reaches crop alpha"):
            normalize_candidate(source, remove_support=False, deskew=False)

    def test_memory_guard_rejects_before_geometry_work(self) -> None:
        # A zero-stride view has a large logical shape without allocating the
        # 324 MB image a copying/geometry operation would otherwise require.
        source = np.broadcast_to(np.zeros((1, 1, 4), dtype=np.uint8), (9000, 9000, 4))
        with patch("scripts.precise_geometry.find_support_cut") as detector:
            with self.assertRaisesRegex(ValueError, "geometry memory guard"):
                normalize_candidate(source)
            detector.assert_not_called()

    def test_memory_guard_accepts_custom_cap_and_explicit_opt_out(self) -> None:
        source = np.zeros((30, 40, 4), dtype=np.uint8)
        source[5:25, 5:35] = (100, 100, 100, 255)
        for limit in (None, "none", 0, 1200, "1,200", "auto"):
            with self.subTest(limit=limit):
                output, info = normalize_candidate(
                    source, remove_support=False, deskew=False,
                    max_geometry_pixels=limit,
                )
                self.assertEqual(output.shape, (20, 30, 4))
                if limit == "auto":
                    self.assertEqual(info["max_geometry_pixels"], DEFAULT_MAX_GEOMETRY_PIXELS)
        with self.assertRaisesRegex(ValueError, "geometry memory guard"):
            normalize_candidate(source, max_geometry_pixels=1199)
        for invalid in (-1, True, "invalid"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    normalize_candidate(source, max_geometry_pixels=invalid)

    def test_support_border_requirement_reaches_original_detector(self) -> None:
        source = synthetic_phone()
        with patch("scripts.precise_geometry.find_support_cut", return_value=None) as detector:
            _, info = normalize_candidate(source, require_support_border=True, deskew=False)
        detector.assert_called_once()
        self.assertTrue(detector.call_args.args[1].require_support_border)
        self.assertTrue(info["require_support_border"])

    def test_invalid_geometry_parameters_fail(self) -> None:
        source = synthetic_phone()
        for keyword, value in (
            ("crop_threshold", 0),
            ("crop_threshold", 255),
            ("crop_threshold", 128.5),
            ("geometry_threshold", True),
            ("confidence_floor", float("nan")),
            ("confidence_floor", 1.1),
            ("max_deskew_degrees", float("inf")),
            ("max_deskew_degrees", -1.0),
        ):
            with self.subTest(keyword=keyword, value=value):
                with self.assertRaises(ValueError):
                    normalize_candidate(source, **{keyword: value})


if __name__ == "__main__":
    unittest.main()
