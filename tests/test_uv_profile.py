#!/usr/bin/env python3
"""Offline numeric-profile, original-pixel, alpha, and bounded-extent tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from phone_normalizer import uv_profile as uv  # noqa: E402


def make_profile(size=(120, 160), bbox=None, crop=None):
    bbox = list(bbox or (0, 0, size[0], size[1]))
    crop = list(crop or (0, 0, size[0], size[1]))
    return {
        "schema_version": 1, "profile_id": "synthetic",
        "body_bbox_method": uv.BODY_BBOX_METHOD,
        "faces": {face: {"reference_size": list(size),
                         "reference_body_bbox": bbox.copy(), "crop_xyxy": crop.copy(),
                         "orientation_quarter_turns": {"left": 1, "right": -1}.get(face, 0)}
                  for face in uv.FACES},
    }


def patterned_image(size):
    w, h = size
    yy, xx = np.mgrid[:h, :w]
    arr = np.empty((h, w, 4), dtype=np.uint8)
    arr[:, :, 0] = ((xx + yy) % 2) * 255
    arr[:, :, 1] = (xx * 17 + yy * 7) % 256
    arr[:, :, 2] = (xx * 3 + yy * 23) % 256
    arr[:, :, 3] = 255
    return Image.fromarray(arr)


class UVProfileTests(unittest.TestCase):
    def test_builtin_numeric_geometry_and_final_sizes(self):
        profile = uv.load_profile("iphone12pro-user-crop-v1")
        expected = {"front": (957, 1985), "back": (960, 1986), "left": (104, 1904),
                    "right": (116, 1877), "top": (1017, 130), "bottom": (1046, 112)}
        self.assertEqual(set(profile["faces"]), set(expected))
        self.assertNotIn("/Users/", json.dumps(profile))
        for face, size in expected.items():
            crop = profile["faces"][face]["crop_xyxy"]
            self.assertEqual((crop[2]-crop[0], crop[3]-crop[1]), size)

    def test_body_edges_ignore_short_camera_bump_and_disconnected_dust(self):
        arr = np.zeros((200, 120, 4), dtype=np.uint8)
        arr[:, 10:100] = (50, 70, 90, 255)
        arr[:25, 100:119] = (50, 70, 90, 255)
        arr[180, 115] = (50, 70, 90, 255)
        self.assertEqual(uv.body_bbox(Image.fromarray(arr)), [10., 0., 100., 200.])
        self.assertEqual(uv.body_bbox(Image.new("RGBA", (1, 1), (1, 2, 3, 255))), [0., 0., 1., 1.])
        with self.assertRaisesRegex(ValueError, "no foreground"):
            uv.body_bbox(Image.new("RGBA", (12, 12)))

    def test_orientation_quarter_turns_are_lossless(self):
        image = patterned_image((7, 4))
        arr = np.asarray(image)
        profile = make_profile()
        for face, turns in (("left", 1), ("right", -1), ("Front", 0)):
            result = uv.orient_candidate(image, face, profile)
            np.testing.assert_array_equal(np.asarray(result), np.rot90(arr, turns))
        np.testing.assert_array_equal(np.asarray(image), arr)
        with self.assertRaises(ValueError):
            uv.orient_candidate(image, "unknown", profile)

    def test_identity_and_crop_preserve_every_rgba_byte(self):
        image = patterned_image((120, 160))
        arr = np.array(image)
        arr[40:48, 35:56, 3] = 23
        arr[51:59, 30:54, 3] = 0  # Invisible RGB also remains byte-identical.
        image = Image.fromarray(arr)
        profile = make_profile(crop=(2, 3, 118, 157))
        with patch.object(uv, "_warp_rgba", side_effect=AssertionError("identity must not resample")):
            result, info = uv.align_to_profile(image, "front", profile)
        np.testing.assert_array_equal(np.asarray(result), arr[3:157, 2:118])
        self.assertEqual(result.size, (116, 154))
        self.assertEqual(info["resampling"], "none_integer_copy")
        self.assertEqual(info["source_resampling_passes_in_final_image"], 0)
        self.assertFalse(info["outer_extent_correction"]["applied"])
        self.assertEqual(info["semantic_anchor_validation"], "not_performed")
        self.assertEqual(info["uv_fit_validation"], "not_performed")

    def test_changed_source_dimensions_retain_profile_crop(self):
        image = patterned_image((70, 180))
        profile = make_profile(bbox=(5, 10, 115, 150), crop=(8, 12, 110, 145))
        result, info = uv.align_to_profile(image, "back", profile)
        self.assertEqual(result.size, (102, 133))
        self.assertEqual(info["crop_xyxy"], [8, 12, 110, 145])
        self.assertEqual(info["source_resampling_passes_in_final_image"], 1)
        self.assertTrue(info["needs_manual_review"])

    def test_premultiplied_sampling_never_leaks_transparent_rgb(self):
        arr = np.zeros((6, 6, 4), dtype=np.uint8)
        arr[:, :, :3] = (0, 0, 255)  # Deliberately poisonous invisible blue.
        arr[1:5, 1:5] = (190, 100, 40, 255)
        image = Image.fromarray(arr)
        profile = make_profile(size=(10, 10), bbox=(2.3, 2.2, 8.1, 8.2), crop=(1, 1, 9, 9))
        result, info = uv.align_to_profile(image, "front", profile)
        out = np.asarray(result)
        visible = out[:, :, 3] > 0
        self.assertTrue(np.any((out[:, :, 3] > 0) & (out[:, :, 3] < 255)))
        self.assertLessEqual(int(np.max(np.abs(out[:, :, :3][visible].astype(int)-[190, 100, 40]))), 1)
        self.assertTrue(np.all(out[:, :, :3][~visible] == 0))
        self.assertEqual(info["source_resampling_passes_in_final_image"], 1)

    def test_one_pixel_extent_uses_bounded_composed_source_sample(self):
        image = patterned_image((116, 220))
        profile = make_profile(size=(116, 220), bbox=(0, 0, 115, 220))
        warp = uv._warp_rgba
        with patch.object(uv, "_warp_rgba", wraps=warp) as calls:
            result, info = uv.align_to_profile(image, "right", profile)
        self.assertTrue(info["outer_extent_correction"]["applied"])
        self.assertAlmostEqual(info["outer_extent_correction"]["axes"]["x"]["scale"], 115/114)
        self.assertEqual(info["remaining_transparent_border_px"], dict(left=0, top=0, right=0, bottom=0))
        self.assertEqual(info["source_resampling_passes_in_final_image"], 1)
        self.assertEqual(calls.call_count, 2)  # Probe, then final sampled from source.
        for call in calls.call_args_list:
            np.testing.assert_array_equal(np.asarray(call.args[0]), np.asarray(image))
        matrix = np.asarray(info["candidate_to_final_matrix"])
        np.testing.assert_array_equal(np.asarray(result), np.asarray(warp(image, matrix, result.size)))
        probe = warp(image, np.asarray(info["candidate_to_reference_matrix"]), result.size)
        twice = warp(probe, np.asarray(info["outer_extent_correction"]["matrix"]), result.size)
        self.assertFalse(np.array_equal(np.asarray(result), np.asarray(twice)))

    def test_large_gap_or_excess_scale_is_not_corrected(self):
        for size, bbox in (((118, 220), (0, 0, 115, 220)), ((60, 90), (0, 0, 59, 90))):
            with self.subTest(size=size):
                image = patterned_image(size)
                profile = make_profile(size=size, bbox=bbox)
                _, info = uv.align_to_profile(image, "top", profile)
                self.assertFalse(info["outer_extent_correction"]["applied"])
                self.assertTrue(info["needs_manual_review"])
                self.assertIn("transparent_outer_border_remains", info["review_reasons"])

    def test_weak_alpha_edge_is_not_treated_as_empty_border(self):
        arr = np.asarray(patterned_image((116, 220))).copy()
        arr[:, -1, 3] = 64
        profile = make_profile(size=(116, 220), bbox=(0, 0, 115, 220))
        result, info = uv.align_to_profile(Image.fromarray(arr), "right", profile)
        self.assertFalse(info["outer_extent_correction"]["applied"])
        np.testing.assert_array_equal(np.asarray(result), arr)

    def test_invalid_profiles_are_rejected(self):
        invalid = []
        profile = make_profile(); del profile["faces"]["top"]; invalid.append(profile)
        profile = make_profile(); profile["faces"]["rear"] = profile["faces"].pop("back"); invalid.append(profile)
        for field, value in (
            ("reference_size", [0, 20]), ("reference_size", [True, 20]),
            ("reference_size", [120., 160]), ("reference_body_bbox", [0, 0, float("nan"), 100]),
            ("reference_body_bbox", [0, 0, float("inf"), 100]),
            ("reference_body_bbox", [5, 0, 5, 100]), ("reference_body_bbox", [0, 0, 121, 100]),
            ("crop_xyxy", [0, 0, 121, 160]), ("crop_xyxy", [0, 0, 0, 160]),
            ("crop_xyxy", [0, 0, 20.5, 30]), ("orientation_quarter_turns", True),
            ("orientation_quarter_turns", 4),
        ):
            profile = make_profile(); profile["faces"]["front"][field] = value; invalid.append(profile)
        profile = make_profile(); profile["schema_version"] = True; invalid.append(profile)
        profile = make_profile(); profile["body_bbox_method"] = "other"; invalid.append(profile)
        profile = make_profile(); profile["extra"] = "unsupported"; invalid.append(profile)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"profile.json"
            path.write_text(json.dumps(make_profile()), encoding="utf-8")
            self.assertEqual(uv.load_profile(path)["profile_id"], "synthetic")
            for profile in invalid:
                with self.subTest(profile=profile):
                    path.write_text(json.dumps(profile), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        uv.load_profile(path)

    def test_memory_guard_rejects_shapes_before_any_image_allocation(self):
        huge = make_profile(size=(9000, 9000))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"huge.json"
            path.write_text(json.dumps(huge), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "geometry memory guard"):
                uv.load_profile(path)
            self.assertEqual(uv.load_profile(path, max_geometry_pixels=None), huge)
        class ShapeOnly:
            size = (9000, 9000)
            def convert(self, mode):
                raise AssertionError("guard must run before conversion")
        with self.assertRaisesRegex(ValueError, "geometry memory guard"):
            uv.align_to_profile(ShapeOnly(), "front", make_profile())
        class TinyShapeOnly(ShapeOnly):
            size = (2, 2)
        with self.assertRaisesRegex(ValueError, "geometry memory guard"):
            uv.align_to_profile(TinyShapeOnly(), "front", huge)
        image = patterned_image((5, 5))
        with self.assertRaisesRegex(ValueError, "geometry memory guard"):
            uv.align_to_profile(image, "front", make_profile(size=(5, 5)), max_geometry_pixels=24)
        result, _ = uv.align_to_profile(image, "front", make_profile(size=(5, 5)), max_geometry_pixels=None)
        self.assertEqual(result.size, (5, 5))


if __name__ == "__main__":
    unittest.main()
