#!/usr/bin/env python3
"""Offline tests for the optional BEN2 geometry stage."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geometry_postprocess import (  # noqa: E402
    DEFAULT_MAX_GEOMETRY_PIXELS,
    GeometryConfig,
    clip_support,
    estimate_long_angle,
    find_support_cut,
    parse_max_geometry_pixels,
    process_array,
    rotate_alpha_safe,
    tight_crop,
    validate_geometry_size,
)


def synthetic_phone() -> np.ndarray:
    """Make a phone with a narrower stand and a fractional alpha edge."""

    size = (320, 480)
    alpha = Image.new("L", size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.rounded_rectangle((55, 28, 264, 350), radius=18, fill=255)
    # The stand is intentionally not centered exactly on the phone.
    draw.rectangle((112, 350, 212, 479), fill=255)
    # A tiny blur emulates BEN2's continuous matte without changing the
    # geometry threshold used by the detector.
    from PIL import ImageFilter

    alpha = alpha.filter(ImageFilter.GaussianBlur(0.7))
    rgb = Image.new("RGB", size, (35, 45, 55))
    rgba = np.dstack([np.asarray(rgb), np.asarray(alpha)])
    return rgba.astype(np.uint8)


def main() -> None:
    # The geometry guard is independent from the model-input cap.  Keep these
    # checks shape-only so the over-limit case never allocates a giant test
    # image merely to verify the early rejection path.
    assert parse_max_geometry_pixels("auto") == DEFAULT_MAX_GEOMETRY_PIXELS
    assert parse_max_geometry_pixels("none") is None
    assert parse_max_geometry_pixels("0") is None
    assert validate_geometry_size((2524, 2524)) == (2524, 2524)
    try:
        validate_geometry_size((9000, 9000))
    except ValueError as exc:
        assert "geometry memory guard" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("an image above the default geometry cap must fail")
    assert GeometryConfig(max_geometry_pixels=None).max_geometry_pixels is None

    rgba = synthetic_phone()
    alpha = rgba[..., 3]
    support = find_support_cut(alpha, GeometryConfig())
    assert support is not None, "synthetic stand should yield a contact chord"
    assert support.threshold_hits >= 2
    clipped = clip_support(alpha, support, confidence_floor=0.0)
    # Pixels well inside the stand are gone, while pixels above the contact
    # remain opaque.  The exact contact y is deliberately not asserted.
    assert int(clipped[430, 160]) == 0
    assert int(clipped[250, 160]) > 200

    result, info = process_array(rgba)
    assert result.shape[2] == 4
    assert result.shape[0] < rgba.shape[0]
    assert result.shape[1] < rgba.shape[1]
    assert info.support is not None
    assert info.support_applied
    assert info.output_size == (result.shape[1], result.shape[0])
    assert info.rotation_degrees == 0.0 or abs(info.rotation_degrees) <= 45.0

    # Alpha-safe rotation must not expose RGB from transparent pixels.
    rotated = rotate_alpha_safe(rgba, 7.0)
    assert rotated.shape[2] == 4
    assert np.all(rotated[..., :3][rotated[..., 3] == 0] == 0)

    cropped, box = tight_crop(rgba)
    assert cropped.shape[2] == 4
    assert box[2] - box[0] == cropped.shape[1]
    assert box[3] - box[1] == cropped.shape[0]

    # A plain rectangle has no support junctions and must be left alone.
    plain = np.zeros((200, 200), dtype=np.uint8)
    plain[30:170, 50:150] = 255
    assert find_support_cut(plain) is None
    plain_angle = float(estimate_long_angle(plain))
    assert min(abs(plain_angle), abs(abs(plain_angle) - 90.0)) < 0.2

    # A camera island can also create two concavities, but it does not extend
    # to the image boundary like the stand in this workflow.  The detector may
    # report a low-confidence diagnostic candidate; the default clip must be a
    # no-op in that case.
    camera = Image.new("L", (320, 480), 0)
    camera_draw = ImageDraw.Draw(camera)
    camera_draw.rounded_rectangle((55, 100, 264, 430), radius=20, fill=255)
    camera_draw.rounded_rectangle((75, 60, 175, 150), radius=12, fill=255)
    camera_alpha = np.asarray(camera)
    camera_cut = find_support_cut(camera_alpha)
    assert camera_cut is not None
    assert camera_cut.confidence < 0.55
    camera_clipped = clip_support(camera_alpha, camera_cut)
    assert np.array_equal(camera_clipped, camera_alpha)

    # The junction is deliberately moved horizontally; no fixed y/x rule is
    # involved in the detector.
    moved = np.zeros((420, 360), dtype=np.uint8)
    cv2.rectangle(moved, (45, 35), (315, 260), 255, -1)
    cv2.rectangle(moved, (145, 260), (218, 419), 255, -1)
    moved_cut = find_support_cut(moved)
    assert moved_cut is not None
    assert moved_cut.confidence >= 0.55

    print("geometry postprocess smoke test passed")


if __name__ == "__main__":
    main()
