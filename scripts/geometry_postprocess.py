#!/usr/bin/env python3
"""Crop and deskew a BiRefNet_dynamic RGBA cutout without changing its edge alpha.

This is deliberately a second stage after BiRefNet_dynamic.  BiRefNet_dynamic supplies the pixels and
the continuous alpha matte; this module only uses a thresholded *copy* of the
alpha channel for geometry.  The output alpha is the original BiRefNet_dynamic alpha,
except for the region confidently identified as the support/stand.

The stand is not located with a fixed y coordinate.  Instead, the largest
alpha component is analysed as a contour.  A stand attached to the phone
usually creates two deep, nearly parallel concave junctions.  The contour path
between the junctions that does not overhang the chord is treated as the stand
branch.  Only the stand side of that chord, in a narrow contact strip, is
removed.  If no reliable pair is found, the alpha is left untouched.

OpenCV is imported lazily so the first-stage BiRefNet_dynamic command remains usable
in environments that do not need this optional geometry stage.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image


# Geometry analysis (connected components, contour fitting and alpha-safe
# rotation) is performed at the source resolution.  A very large source can
# therefore require several temporary arrays even though the segmentation
# model itself is capped at 1024 px.  Keep a conservative default guard so an
# accidental panorama/scan cannot exhaust a Mac's unified memory.  The limit
# is deliberately expressed in pixels rather than a side length so ordinary
# wide or tall product images are not rejected unnecessarily.
DEFAULT_MAX_GEOMETRY_PIXELS = 64_000_000  # 64 megapixels


def parse_max_geometry_pixels(value: object) -> Optional[int]:
    """Parse the geometry memory guard value.

    ``"auto"``/``"default"`` select the project default, while ``None`` or
    ``"none"``/``0`` explicitly disable the guard for a controlled caller.
    Keeping this parser permissive preserves the convenient string values used
    by the command-line interfaces and the typed ``None`` API idiom.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            "max_geometry_pixels must be a positive integer, 'auto', or 'none'"
        )
    if isinstance(value, (int, np.integer)):
        pixels = int(value)
    else:
        text = str(value).strip().lower().replace(",", "")
        if text in {"", "auto", "default"}:
            return DEFAULT_MAX_GEOMETRY_PIXELS
        if text in {"none", "off", "unlimited", "0"}:
            return None
        if not text.isdigit():
            raise ValueError(
                "max_geometry_pixels must be a positive integer, 'auto', or 'none'"
            )
        pixels = int(text)
    if pixels == 0:
        return None
    if pixels < 1:
        raise ValueError("max_geometry_pixels must be positive")
    return pixels


def validate_geometry_size(
    image_size: tuple[int, int],
    max_geometry_pixels: object = DEFAULT_MAX_GEOMETRY_PIXELS,
) -> tuple[int, int]:
    """Validate a source ``(width, height)`` against the geometry guard.

    The check happens before any full-resolution geometry scratch buffers are
    allocated.  It raises ``ValueError`` (an input/configuration problem, not
    an allocator failure) with an actionable opt-out hint.
    """

    if len(image_size) != 2:
        raise ValueError("image_size must contain width and height")
    width, height = int(image_size[0]), int(image_size[1])
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    cap = parse_max_geometry_pixels(max_geometry_pixels)
    pixels = width * height
    if cap is not None and pixels > cap:
        raise ValueError(
            "source image "
            f"{width}×{height} ({pixels / 1_000_000:.1f} MP) exceeds the "
            f"geometry memory guard ({cap / 1_000_000:.1f} MP). "
            "Resize the source, lower the geometry workload, or explicitly "
            "set max_geometry_pixels=None/--max-geometry-pixels none if the "
            "machine has enough memory."
        )
    return width, height


def _cv2():
    """Import OpenCV with a useful error for users of the optional stage."""

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "geometry post-processing requires opencv-python-headless; "
            "install the project requirements first"
        ) from exc
    return cv2


@dataclass(frozen=True)
class GeometryConfig:
    """Tunable geometry thresholds, expressed mostly relative to image size."""

    geometry_threshold: int = 32
    consistency_thresholds: tuple[int, ...] = (32, 64, 128, 192)
    contour_epsilon_fraction: float = 0.005
    min_concavity_sine: float = 0.08
    min_purity: float = 0.80
    max_overhang_ratio: float = 0.20
    axis_tolerance_degrees: float = 15.0
    min_component_area_fraction: float = 0.002
    contact_margin_fraction: float = 0.12
    min_contact_margin: float = 8.0
    crop_threshold: int = 8
    # Roll is normally small in the catalog shots, but there is no reason to
    # silently leave a clean 20–30° in-plane tilt.  Callers that only want
    # micro-corrections can pass a smaller bound.
    max_deskew_degrees: float = 45.0
    # When enabled, reject candidates whose stand branch does not reach the
    # image boundary.  The default keeps this as a soft prior so callers can
    # lower ``confidence_floor`` for a floating stand or provide manual hints.
    require_support_border: bool = False
    min_edge_reach: float = 0.12
    # Full-resolution geometry is retained for normal product photos.  The
    # guard only rejects unusually large inputs before scratch arrays are made;
    # ``None`` (or ``"none"`` through a CLI/config adapter) opts out.
    max_geometry_pixels: Optional[int] = DEFAULT_MAX_GEOMETRY_PIXELS

    def __post_init__(self) -> None:
        parsed = parse_max_geometry_pixels(self.max_geometry_pixels)
        object.__setattr__(self, "max_geometry_pixels", parsed)


@dataclass(frozen=True)
class SupportCut:
    """A detected phone/support contact chord in image coordinates."""

    p: tuple[float, float]
    q: tuple[float, float]
    support_sign: float
    length: float
    depth: float
    purity: float
    overhang_ratio: float
    axis_residual: float
    score: float
    confidence: float
    threshold_hits: int = 1
    edge_reach: float = 0.0

    @property
    def tangent(self) -> np.ndarray:
        d = np.asarray(self.q, dtype=np.float64) - np.asarray(self.p, dtype=np.float64)
        n = np.linalg.norm(d)
        return d / n if n else np.array([1.0, 0.0])

    @property
    def normal(self) -> np.ndarray:
        t = self.tangent
        return np.array([-t[1], t[0]], dtype=np.float64)


@dataclass(frozen=True)
class GeometryResult:
    """Diagnostics returned by :func:`process` in addition to the image."""

    support: Optional[SupportCut]
    support_applied: bool
    angle_degrees: Optional[float]
    rotation_degrees: float
    crop_box: tuple[int, int, int, int]
    output_size: tuple[int, int]


def _largest_component(mask: np.ndarray, min_area: int = 1) -> np.ndarray:
    """Return the largest connected component of a boolean mask."""

    cv2 = _cv2()
    mask8 = (np.asarray(mask) > 0).astype(np.uint8)
    if not np.any(mask8):
        return np.zeros_like(mask8, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask8, 8)
    ids = [
        i
        for i in range(1, n)
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area)
    ]
    if not ids:
        return np.zeros_like(mask8, dtype=bool)
    chosen = max(ids, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    return labels == chosen


def _contour(mask: np.ndarray) -> Optional[np.ndarray]:
    cv2 = _cv2()
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


def _normalize_line_angle(angle: float) -> float:
    """Normalize an unoriented image line angle to [-90, 90)."""

    return ((float(angle) + 90.0) % 180.0) - 90.0


def _project(points: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Project Nx2 points without a large BLAS matmul.

    Some NumPy 2/OpenBLAS builds on macOS emit spurious floating-point
    warnings (and can return corrupted values) for very large ``N x 2 @ 2``
    products.  Elementwise projection is just as fast for this small vector
    and is deterministic across those builds.
    """

    points = np.asarray(points, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    return points[..., 0] * vector[0] + points[..., 1] * vector[1]


def _angular_distance(a: float, b: float) -> float:
    return abs(_normalize_line_angle(float(a) - float(b)))


def _long_axis(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return long and cross unit vectors plus the long-axis angle."""

    cv2 = _cv2()
    rect = cv2.minAreaRect(contour.astype(np.float32).reshape(-1, 1, 2))
    box = cv2.boxPoints(rect).astype(np.float64)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    t = edges[int(np.argmax(lengths))]
    t /= max(float(np.linalg.norm(t)), 1e-12)
    # An unoriented axis is sufficient; choose a deterministic sign.
    if t[0] < 0 or (abs(t[0]) < 1e-12 and t[1] < 0):
        t = -t
    angle = _normalize_line_angle(math.degrees(math.atan2(float(t[1]), float(t[0]))))
    n = np.array([-t[1], t[0]], dtype=np.float64)
    return t, n, angle


def _cyclic_path(poly: np.ndarray, start: int, end: int) -> np.ndarray:
    out: list[int] = []
    k = int(start)
    while True:
        out.append(k)
        if k == end:
            break
        k = (k + 1) % len(poly)
    return poly[np.asarray(out, dtype=np.int64)]


def _candidate_paths(
    poly: np.ndarray,
    i: int,
    j: int,
    image_shape: tuple[int, int],
    body_axis: float,
    body_cross_axis: float,
    config: GeometryConfig,
) -> list[dict]:
    """Build both contour-path interpretations for a concave pair."""

    cv2 = _cv2()
    p, q = poly[i], poly[j]
    delta = q - p
    length = float(np.linalg.norm(delta))
    if length < 20.0:
        return []
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    h, w = image_shape
    edge_margin = max(2.0, 0.006 * min(h, w))
    result: list[dict] = []
    for start, end in ((i, j), (j, i)):
        points = _cyclic_path(poly, start, end)
        relative = points - p
        projections = _project(relative, tangent)
        signed = _project(relative, normal)
        interior = signed[1:-1]
        if len(interior) == 0:
            continue
        median_signed = float(np.median(interior))
        sign = 1.0 if median_signed >= 0.0 else -1.0
        purity = float(np.mean(np.sign(interior) == sign))
        overhang = max(0.0, float(-projections.min())) + max(
            0.0, float(projections.max() - length)
        )
        overhang_ratio = overhang / length
        depth = float(np.max(np.abs(interior)))
        # The support branch is normally a narrow path leaving the chord.  Its
        # area is useful, but a long phone perimeter can have a larger area, so
        # overhang and border reach are used as independent safeguards.
        branch_area = abs(
            float(
                cv2.contourArea(
                    np.vstack([p, points[1:]])
                    .astype(np.float32)
                    .reshape(-1, 1, 2)
                )
            )
        )
        near_edge = (
            (points[:, 0] <= edge_margin)
            | (points[:, 0] >= w - 1.0 - edge_margin)
            | (points[:, 1] <= edge_margin)
            | (points[:, 1] >= h - 1.0 - edge_margin)
        )
        edge_reach = float(np.mean(near_edge))
        # Contact lines should be parallel to one of the phone axes.  Keep the
        # value in the diagnostic even when it fails the hard gate.
        line_angle = math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))
        axis_residual = min(
            _angular_distance(line_angle, body_axis),
            _angular_distance(line_angle, body_cross_axis),
        )
        # Depth normalized by chord length makes this resolution-independent.
        score = branch_area * (1.0 + depth / max(length, 1.0))
        score *= math.exp(-((axis_residual / 6.0) ** 2))
        # A branch touching a frame is a strong signal, but do not require it:
        # a stand can end before the photograph boundary.
        score *= 1.0 + 0.75 * edge_reach
        result.append(
            {
                "p": p.copy(),
                "q": q.copy(),
                "tangent": tangent.copy(),
                "normal": normal.copy(),
                "signed": signed.copy(),
                "support_sign": sign,
                "length": length,
                "purity": purity,
                "overhang_ratio": overhang_ratio,
                "depth": depth,
                "area": branch_area,
                "edge_reach": edge_reach,
                "axis_residual": axis_residual,
                "score": score,
                "start": start,
            }
        )
    return result


def _find_support_single(
    alpha: np.ndarray, threshold: int, config: GeometryConfig
) -> Optional[dict]:
    cv2 = _cv2()
    alpha = np.asarray(alpha)
    h, w = alpha.shape
    # Keep a small absolute floor to reject isolated matte specks, but do not
    # discard the entire subject in thumbnail-sized inputs.
    min_area = max(16, int(h * w * config.min_component_area_fraction))
    geometry = _largest_component(alpha >= threshold, min_area=min_area)
    contour = _contour(geometry)
    if contour is None:
        return None
    perimeter = float(cv2.arcLength(contour.astype(np.float32).reshape(-1, 1, 2), True))
    epsilon = max(2.0, config.contour_epsilon_fraction * perimeter)
    poly = cv2.approxPolyDP(
        contour.astype(np.float32).reshape(-1, 1, 2), epsilon, True
    ).reshape(-1, 2).astype(np.float64)
    if len(poly) < 4:
        return None

    body_t, body_n, body_axis = _long_axis(contour)
    body_cross_axis = _normalize_line_angle(body_axis + 90.0)
    oriented_area = float(
        cv2.contourArea(poly.astype(np.float32).reshape(-1, 1, 2), oriented=True)
    )
    orientation = 1.0 if oriented_area >= 0.0 else -1.0
    concave: list[int] = []
    for k in range(len(poly)):
        prev = poly[k - 1] - poly[k]
        nxt = poly[(k + 1) % len(poly)] - poly[k]
        cross = float(prev[0] * nxt[1] - prev[1] * nxt[0])
        scale = float(np.linalg.norm(prev) * np.linalg.norm(nxt))
        if scale > 1e-6 and orientation * cross / scale > config.min_concavity_sine:
            concave.append(k)

    candidates: list[dict] = []
    for i, j in itertools.combinations(concave, 2):
        candidates.extend(
            _candidate_paths(
                poly,
                i,
                j,
                (h, w),
                body_axis,
                body_cross_axis,
                config,
            )
        )
    accepted = [
        c
        for c in candidates
        if c["purity"] >= config.min_purity
        and c["overhang_ratio"] <= config.max_overhang_ratio
        and c["axis_residual"] <= config.axis_tolerance_degrees
    ]
    if not accepted:
        return None
    return max(accepted, key=lambda c: float(c["score"]))


def _endpoint_distance(a: dict, b: dict) -> float:
    p1, q1 = np.asarray(a["p"]), np.asarray(a["q"])
    p2, q2 = np.asarray(b["p"]), np.asarray(b["q"])
    return float(min(np.linalg.norm(p1 - p2) + np.linalg.norm(q1 - q2), np.linalg.norm(p1 - q2) + np.linalg.norm(q1 - p2)))


def find_support_cut(
    alpha: np.ndarray, config: GeometryConfig = GeometryConfig()
) -> Optional[SupportCut]:
    """Find a support contact line, requiring agreement across alpha levels.

    The consensus step prevents a single noisy antialiased contour from
    creating a destructive cut.  A candidate is accepted when at least half of
    the configured thresholds agree within a scale-relative endpoint distance.
    """

    alpha = np.asarray(alpha)
    if alpha.ndim != 2:
        raise ValueError("alpha must be a 2-D uint8 array")
    raw: list[dict] = []
    for threshold in config.consistency_thresholds:
        candidate = _find_support_single(alpha, int(threshold), config)
        if candidate is not None:
            candidate = dict(candidate)
            candidate["threshold"] = int(threshold)
            raw.append(candidate)
    if not raw:
        return None
    # Cluster by endpoint proximity.  The tolerance scales with the image, not
    # with an absolute pixel count, so phone photos of different resolutions
    # behave consistently.
    scale = max(1.0, float(min(alpha.shape)))
    tolerance = max(5.0, 0.012 * scale)
    clusters: list[list[dict]] = []
    for candidate in raw:
        cluster = next(
            (group for group in clusters if _endpoint_distance(candidate, group[0]) <= tolerance),
            None,
        )
        if cluster is None:
            clusters.append([candidate])
        else:
            cluster.append(candidate)
    cluster = max(clusters, key=lambda group: (len(group), sum(c["score"] for c in group)))
    required = max(2, int(math.ceil(len(config.consistency_thresholds) / 2)))
    if len(cluster) < required:
        return None
    # Align endpoint order to the first candidate, then take medians.  This
    # retains subpixel-ish stability while ignoring threshold-specific contour
    # stair-steps.
    reference = cluster[0]
    rp, rq = np.asarray(reference["p"]), np.asarray(reference["q"])
    ps: list[np.ndarray] = []
    qs: list[np.ndarray] = []
    for candidate in cluster:
        p, q = np.asarray(candidate["p"]), np.asarray(candidate["q"])
        if np.linalg.norm(p - rp) + np.linalg.norm(q - rq) <= np.linalg.norm(p - rq) + np.linalg.norm(q - rp):
            ps.append(p)
            qs.append(q)
        else:
            ps.append(q)
            qs.append(p)
    p = np.median(np.stack(ps), axis=0)
    q = np.median(np.stack(qs), axis=0)
    # Use the candidate with the strongest score for side/depth diagnostics.
    best = max(cluster, key=lambda c: float(c["score"]))
    # ``p``/``q`` may have been reversed while aligning the threshold
    # candidates.  The normal (and therefore the sign identifying the support
    # side) must follow the final median endpoint order.
    bp, bq = np.asarray(best["p"]), np.asarray(best["q"])
    best_reversed = np.linalg.norm(bp - p) + np.linalg.norm(bq - q) > np.linalg.norm(bp - q) + np.linalg.norm(bq - p)
    # Confidence is intentionally conservative.  Callers can expose it to a
    # UI and ask for manual points when it falls below their policy threshold.
    alignment = max(0.0, 1.0 - float(best["axis_residual"]) / 15.0)
    purity = max(0.0, min(1.0, float(best["purity"])))
    consensus = min(1.0, len(cluster) / max(len(config.consistency_thresholds), 1))
    overhang = max(0.0, 1.0 - float(best["overhang_ratio"]) / max(config.max_overhang_ratio, 1e-6))
    # A contour branch that reaches (or comes very close to) the photograph
    # boundary is characteristic of the display stand in this workflow.  A
    # camera island can have two convincing concavities but normally has no
    # such extension; down-weighting it keeps the default operation safe.  We
    # retain a soft term rather than making a hard border requirement, so a
    # floating stand can still be enabled explicitly with a lower floor.
    edge_reach = float(best.get("edge_reach", 0.0))
    edge_signal = min(1.0, edge_reach / max(config.min_edge_reach, 1e-6))
    if config.require_support_border and edge_signal < 1.0:
        return None
    confidence = 0.35 * consensus + 0.25 * alignment + 0.20 * purity + 0.20 * overhang
    # Keep a meaningful (but deliberately sub-default) confidence for a
    # floating support.  This lets an application opt in with an explicit
    # lower floor while preventing an automatic cut in the default pipeline.
    confidence *= 0.35 + 0.65 * edge_signal
    return SupportCut(
        p=(float(p[0]), float(p[1])),
        q=(float(q[0]), float(q[1])),
        support_sign=float(-best["support_sign"] if best_reversed else best["support_sign"]),
        length=float(np.linalg.norm(q - p)),
        depth=float(best["depth"]),
        purity=float(best["purity"]),
        overhang_ratio=float(best["overhang_ratio"]),
        axis_residual=float(best["axis_residual"]),
        score=float(best["score"]),
        confidence=float(confidence),
        threshold_hits=len(cluster),
        edge_reach=edge_reach,
    )


def clip_support(
    alpha: np.ndarray,
    support: Optional[SupportCut],
    confidence_floor: float = 0.55,
    margin_fraction: float = 0.12,
    min_margin: float = 8.0,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Remove only the support side of a contact strip from ``alpha``.

    This function never touches RGB.  A low-confidence detection is a no-op,
    which is safer than cutting a legitimate phone corner or camera bump.
    """

    alpha_array = np.asarray(alpha)
    if alpha_array.ndim != 2:
        raise ValueError("alpha must be a 2-D array")
    # The public/default path owns its result, while ``process_array`` can
    # explicitly pass ``copy=False`` after it has taken ownership of the RGBA
    # buffer.  Avoiding this otherwise redundant full-frame alpha copy keeps
    # the geometry stage's peak bounded for large (but still accepted) photos.
    out = alpha_array.copy() if copy else alpha_array
    if not out.flags.writeable:
        out = out.copy()
    if support is None or support.confidence < confidence_floor:
        return out
    h, w = out.shape
    p = np.asarray(support.p, dtype=np.float32)
    # A full-frame float64 mesh here used to allocate six additional arrays
    # (yy/xx, dx/dy, projection/signed).  At 10K×10K that temporary alone can
    # exceed several GB and could undo the model input cap.  Evaluate the
    # contact strip in bounded row chunks instead; the result is identical up
    # to sub-pixel float precision while peak memory depends on chunk width,
    # not on the total image height.
    t = np.asarray(support.tangent, dtype=np.float32)
    n = np.asarray(support.normal, dtype=np.float32)
    margin = max(float(min_margin), float(margin_fraction) * support.length)
    x = np.arange(w, dtype=np.float32)[None, :]
    # Keep each working slab comfortably below a few MB for ordinary camera
    # widths.  A very wide panorama is still bounded by a smaller slab rather
    # than multiplying the chunk size by its width.
    target_elements = 1_000_000
    chunk_rows = max(1, min(h, target_elements // max(w, 1)))
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        y = np.arange(y0, y1, dtype=np.float32)[:, None]
        dx = x - p[0]
        dy = y - p[1]
        projection = dx * t[0] + dy * t[1]
        strip = (projection >= -margin) & (projection <= support.length + margin)
        del projection
        signed = dx * n[0] + dy * n[1]
        remove = strip & (float(support.support_sign) * signed >= 0.0)
        out[y0:y1][remove] = 0
        del dx, dy, signed, strip, remove, y
    del x
    return out


def _main_geometry(alpha: np.ndarray, threshold: int) -> np.ndarray:
    h, w = alpha.shape
    return _largest_component(
        alpha >= threshold,
        min_area=max(1, int(h * w * 0.001)),
    )


def _fit_profile_line(points: np.ndarray, residual_limit: float) -> Optional[tuple[float, float, float]]:
    """Robustly fit an unoriented line; return (angle, inlier_fraction, span)."""

    if len(points) < 12:
        return None
    points = np.asarray(points, dtype=np.float64)
    # Use deterministic RANSAC.  The point set is a boundary profile, so two
    # randomly selected points define a candidate line well and outliers are
    # usually rounded corners/camera protrusions.
    rng = np.random.default_rng(12345)
    best: Optional[tuple[int, float, np.ndarray]] = None
    for _ in range(min(300, max(80, len(points) * 2))):
        i, j = rng.choice(len(points), 2, replace=False)
        d = points[j] - points[i]
        norm = float(np.linalg.norm(d))
        if norm < 1.0:
            continue
        normal = np.array([-d[1], d[0]], dtype=np.float64) / norm
        residual = np.abs(_project(points - points[i], normal))
        inlier = residual <= residual_limit
        count = int(np.count_nonzero(inlier))
        med = float(np.median(residual[inlier])) if count else float("inf")
        if best is None or (count, -med) > (best[0], -best[1]):
            best = (count, med, inlier)
    if best is None or best[0] < max(8, int(0.20 * len(points))):
        return None
    inliers = points[best[2]]
    # Orthogonal least squares on RANSAC inliers.
    center = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - center, full_matrices=False)
    direction = vt[0]
    angle = _normalize_line_angle(math.degrees(math.atan2(float(direction[1]), float(direction[0]))))
    span = float(np.ptp(_project(inliers - center, direction)))
    return angle, float(best[0] / len(points)), span


def _axis_aligned_profiles(
    geometry: np.ndarray,
    horizontal: bool,
    quantile: tuple[float, float] = (0.08, 0.92),
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two boundary profiles of an almost axis-aligned object.

    For a horizontal phone, each column contributes its topmost and bottommost
    foreground pixels.  For a vertical phone, each row contributes its leftmost
    and rightmost pixels.  Keeping the profiles separate is important: a
    camera bump usually contaminates only the top profile, while the opposite
    long edge remains a clean deskew reference.
    """

    ys, xs = np.where(geometry)
    if len(xs) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    lo, hi = np.quantile(xs if horizontal else ys, quantile)
    first: list[tuple[float, float]] = []
    second: list[tuple[float, float]] = []
    if horizontal:
        for x in range(int(math.ceil(lo)), int(math.floor(hi)) + 1):
            yy = np.flatnonzero(geometry[:, x])
            if len(yy):
                first.append((float(x), float(yy.min())))
                second.append((float(x), float(yy.max())))
    else:
        for y in range(int(math.ceil(lo)), int(math.floor(hi)) + 1):
            xx = np.flatnonzero(geometry[y, :])
            if len(xx):
                first.append((float(xx.min()), float(y)))
                second.append((float(xx.max()), float(y)))
    return np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)


def estimate_long_angle(alpha: np.ndarray, threshold: int = 32) -> Optional[float]:
    """Estimate the phone long-edge angle in image coordinates.

    ``minAreaRect`` only supplies the coordinate frame.  The actual angle is
    fitted to the long boundary profiles, which avoids a camera bump bias.  A
    horizontal phone uses per-column bottom/top profiles; a vertical phone uses
    per-row left/right profiles.  The most linear side(s) win.
    """

    cv2 = _cv2()
    alpha = np.asarray(alpha)
    geometry = _main_geometry(alpha, threshold)
    contour = _contour(geometry)
    if contour is None:
        return None
    t, n, frame_angle = _long_axis(contour)
    ys, xs = np.where(geometry)
    if len(xs) < 20:
        return None

    # The two supplied orientations (and most catalog shots) are close to
    # horizontal or vertical.  Axis-aligned profiles are less biased than a
    # rotated projection when a camera island protrudes from one side.  Keep a
    # generic projected fallback below for genuinely diagonal photos.
    residual_limit = max(2.0, 0.004 * min(alpha.shape))
    if abs(frame_angle) <= 20.0:
        top, bottom = _axis_aligned_profiles(geometry, horizontal=True)
        fits = [
            _fit_profile_line(top, residual_limit),
            _fit_profile_line(bottom, residual_limit),
        ]
        # Prefer the lower edge when it is coherent; it is not affected by the
        # camera island in the tested rear-phone views.  Fall back to the best
        # side when a support or crop hides that edge.
        if fits[1] is not None and fits[1][1] >= 0.35 and fits[1][2] >= 0.45 * max(1.0, abs(float(xs.max() - xs.min()))):
            return _normalize_line_angle(fits[1][0])
        valid = [fit for fit in fits if fit is not None]
        if valid:
            best = max(valid, key=lambda fit: fit[1] * fit[2])
            return _normalize_line_angle(best[0])
    elif abs(abs(frame_angle) - 90.0) <= 20.0:
        left, right = _axis_aligned_profiles(geometry, horizontal=False)
        fits = [
            _fit_profile_line(left, residual_limit),
            _fit_profile_line(right, residual_limit),
        ]
        valid = [fit for fit in fits if fit is not None]
        if valid:
            # Average the two unoriented line angles.  If one side is much
            # shorter, use the longer coherent side only.
            strongest = max(valid, key=lambda fit: fit[1] * fit[2])
            chosen = [
                fit
                for fit in valid
                if fit[1] >= 0.45 and fit[2] >= 0.35 * strongest[2]
            ] or [strongest]
            doubled = np.radians([fit[0] for fit in chosen]) * 2.0
            avg = 0.5 * math.degrees(
                math.atan2(float(np.mean(np.sin(doubled))), float(np.mean(np.cos(doubled))))
            )
            return _normalize_line_angle(avg)

    # Project all foreground pixels into the frame.  Quantizing the long-axis
    # coordinate into integer bins makes this insensitive to tiny mask gaps.
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = coords.mean(axis=0)
    relative = coords - center
    u = _project(relative, t)
    v = _project(relative, n)
    lo, hi = np.quantile(u, [0.05, 0.95])
    keep = (u >= lo) & (u <= hi)
    u, v, coords = u[keep], v[keep], coords[keep]
    if len(u) < 40:
        return frame_angle
    bins = np.floor(u).astype(np.int64)
    profiles: list[np.ndarray] = []
    # For each long-axis bin, retain both cross-axis extremes.  A camera bump
    # tends to affect only a short run and is removed by RANSAC.
    for b in np.unique(bins):
        sel = bins == b
        if np.count_nonzero(sel) < 2:
            continue
        for extreme in (np.min(v[sel]), np.max(v[sel])):
            idx = np.flatnonzero(sel)[int(np.argmin(np.abs(v[sel] - extreme)))]
            profiles.append(coords[idx])
    if len(profiles) < 20:
        return frame_angle
    points = np.asarray(profiles, dtype=np.float64)
    # Split sides by cross-axis sign and fit independently.  This avoids one
    # side's outliers dominating the other.
    side_results: list[tuple[float, float, float, float]] = []
    pv = _project(points - center, n)
    for side in (-1.0, 1.0):
        side_points = points[(side * pv) >= np.quantile(side * pv, 0.70)]
        fit = _fit_profile_line(side_points, residual_limit=max(2.0, 0.006 * min(alpha.shape)))
        if fit is not None:
            angle, inlier_fraction, span = fit
            residual = _angular_distance(angle, frame_angle)
            side_results.append((angle, inlier_fraction, span, residual))
    if not side_results:
        return frame_angle
    # Weight long, coherent fits.  If both sides exist, circular averaging is
    # stable; otherwise use the best side.
    side_results.sort(key=lambda x: (x[1] * x[2], -x[3]), reverse=True)
    best = side_results[0]
    chosen = [x for x in side_results if x[1] >= 0.45 and x[2] >= 0.35 * best[2]]
    if not chosen:
        chosen = [best]
    # Unoriented angles need doubled-angle averaging.
    angles = np.radians([x[0] for x in chosen])
    avg = 0.5 * math.degrees(math.atan2(float(np.mean(np.sin(2 * angles))), float(np.mean(np.cos(2 * angles)))))
    return _normalize_line_angle(avg)


def rotate_alpha_safe(rgba: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate premultiplied RGB and alpha separately to avoid matte fringes."""

    image = np.asarray(rgba)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("rgba must have shape (height, width, 4)")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if abs(float(degrees)) < 1e-5:
        # Keep transparent pixels colour-clean even when no interpolation is
        # needed.  RGB under alpha==0 is not visible, but leaving source
        # colours there can surprise downstream compositors and makes it
        # possible to re-introduce a halo if the alpha is edited later.
        result = image.copy()
        result[..., :3][result[..., 3] == 0] = 0
        return result
    alpha = image[..., 3].astype(np.float32)
    premultiplied = np.rint(image[..., :3].astype(np.float32) * alpha[..., None] / 255.0).clip(0, 255).astype(np.uint8)
    rgb_rot = Image.fromarray(premultiplied).rotate(
        float(degrees), resample=Image.Resampling.BICUBIC, expand=True
    )
    alpha_rot = Image.fromarray(image[..., 3].astype(np.uint8)).rotate(
        float(degrees), resample=Image.Resampling.BICUBIC, expand=True
    )
    rgb = np.asarray(rgb_rot).astype(np.float32)
    a = np.asarray(alpha_rot).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = np.where(a[..., None] > 1.0, rgb * 255.0 / a[..., None], 0.0)
    result = np.dstack(
        [rgb.clip(0, 255).astype(np.uint8), a.clip(0, 255).astype(np.uint8)]
    )
    result[..., :3][result[..., 3] == 0] = 0
    return result


def tight_crop(
    rgba: np.ndarray,
    threshold: int = 8,
    *,
    copy: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Tight-crop the largest alpha component, preserving interior alpha."""

    raw = np.asarray(rgba)
    image = raw.copy() if copy else raw
    if not image.flags.writeable:
        image = image.copy()
    geometry = _main_geometry(image[..., 3], threshold)
    ys, xs = np.where(geometry)
    if len(xs) == 0:
        return image, (0, 0, image.shape[1], image.shape[0])
    # Remove interpolation specks only outside the selected component.  Alpha
    # values *inside* the component remain exactly as supplied by BiRefNet_dynamic.
    image[..., 3][~geometry] = 0
    image[..., :3][image[..., 3] == 0] = 0
    box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1], box


def process_array(
    rgba: np.ndarray,
    config: GeometryConfig = GeometryConfig(),
    confidence_floor: float = 0.55,
    *,
    remove_support: bool = True,
    deskew: bool = True,
    tight_crop_enabled: bool = True,
    copy: bool = True,
) -> tuple[np.ndarray, GeometryResult]:
    """Process an in-memory segmented RGBA image.

    ``remove_support``, ``deskew`` and ``tight_crop_enabled`` are explicit
    switches so a caller can compare each stage without changing the
    segmentation backend.  Geometry diagnostics are still computed when
    deskew is disabled (the reported angle is useful in metadata), but no
    rotation is applied.  The continuous input alpha is preserved everywhere
    except the confidently identified support contact strip.
    """

    raw = np.asarray(rgba)
    if raw.ndim != 3 or raw.shape[2] != 4:
        raise ValueError("rgba must have shape (height, width, 4)")
    # A pure pass-through does not allocate the geometry scratch buffers, so
    # preserve the historical ability to use it with an explicitly huge image.
    # All normal/default paths pass through this guard before the first full
    # RGBA copy, preventing a geometry OOM from taking down the host process.
    if remove_support or deskew or tight_crop_enabled:
        validate_geometry_size(
            (int(raw.shape[1]), int(raw.shape[0])),
            config.max_geometry_pixels,
        )
    # ``copy=True`` preserves the historical non-mutating contract.  Internal
    # callers that already own a freshly materialized RGBA buffer can opt into
    # ``copy=False`` to avoid a second 4-byte-per-pixel allocation.
    image = raw.copy() if copy else raw
    if not image.flags.writeable:
        image = image.copy()
    if image.dtype != np.uint8:
        image = image.clip(0, 255).astype(np.uint8)
    # With every geometry stage explicitly disabled, honour the caller's
    # request for a true pass-through.  In particular, do not threshold away
    # low-alpha BiRefNet_dynamic fringe pixels merely to compute a component that will not
    # be used.
    if not remove_support and not deskew and not tight_crop_enabled:
        image[..., :3][image[..., 3] == 0] = 0
        result = GeometryResult(
            support=None,
            support_applied=False,
            angle_degrees=None,
            rotation_degrees=0.0,
            crop_box=(0, 0, int(image.shape[1]), int(image.shape[0])),
            output_size=(int(image.shape[1]), int(image.shape[0])),
        )
        return image, result
    support = find_support_cut(image[..., 3], config) if remove_support else None
    clipped_alpha = clip_support(
        image[..., 3],
        support,
        confidence_floor=confidence_floor,
        margin_fraction=config.contact_margin_fraction,
        min_margin=config.min_contact_margin,
        copy=False,
    )
    support_applied = bool(
        support is not None and support.confidence >= float(confidence_floor)
    )
    # Remove tiny frame specks before interpolation, without touching the
    # continuous matte in the principal component.
    principal = _main_geometry(clipped_alpha, config.geometry_threshold)
    clipped_alpha[~principal] = 0
    image[..., 3] = clipped_alpha
    # The source image can contain arbitrary RGB in regions BiRefNet_dynamic marked
    # transparent.  Clear it before the optional rotation so the no-rotation
    # path and the rotated path have identical, compositor-safe semantics.
    image[..., :3][clipped_alpha == 0] = 0
    angle = estimate_long_angle(clipped_alpha, config.geometry_threshold)
    if angle is None or not deskew:
        rotation = 0.0
    else:
        target = 0.0 if abs(angle) <= 45.0 else (90.0 if angle >= 0 else -90.0)
        rotation = float(angle - target)
        if abs(rotation) > config.max_deskew_degrees:
            rotation = 0.0
    rotated = rotate_alpha_safe(image, rotation)
    if tight_crop_enabled:
        cropped, box = tight_crop(rotated, config.crop_threshold, copy=False)
    else:
        cropped = rotated
        box = (0, 0, int(rotated.shape[1]), int(rotated.shape[0]))
    result = GeometryResult(
        support=support,
        support_applied=support_applied,
        angle_degrees=None if angle is None else float(angle),
        rotation_degrees=float(rotation),
        crop_box=box,
        output_size=(int(cropped.shape[1]), int(cropped.shape[0])),
    )
    return cropped, result


def process(
    input_path: Path,
    output_path: Path,
    config: GeometryConfig = GeometryConfig(),
    confidence_floor: float = 0.55,
    *,
    remove_support: bool = True,
    deskew: bool = True,
    tight_crop_enabled: bool = True,
) -> GeometryResult:
    """Read a BiRefNet_dynamic PNG, process it, and write a transparent PNG."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    with Image.open(input_path) as opened:
        rgba = np.asarray(opened.convert("RGBA"))
    output, result = process_array(
        rgba,
        config,
        confidence_floor=confidence_floor,
        remove_support=remove_support,
        deskew=deskew,
        tight_crop_enabled=tight_crop_enabled,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(output_path, format="PNG")
    return result


def _result_json(result: GeometryResult) -> dict:
    support = result.support
    return {
        "support": None
        if support is None
        else {
            "p": list(support.p),
            "q": list(support.q),
            "confidence": support.confidence,
            "threshold_hits": support.threshold_hits,
            "axis_residual": support.axis_residual,
            "overhang_ratio": support.overhang_ratio,
            "edge_reach": support.edge_reach,
        },
        "support_applied": result.support_applied,
        "angle_degrees": result.angle_degrees,
        "rotation_degrees": result.rotation_degrees,
        "crop_box": list(result.crop_box),
        "output_size": list(result.output_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="BiRefNet_dynamic RGBA PNG")
    parser.add_argument("output", type=Path, help="processed transparent PNG")
    parser.add_argument("--confidence-floor", type=float, default=0.55)
    parser.add_argument(
        "--max-geometry-pixels",
        default=str(DEFAULT_MAX_GEOMETRY_PIXELS),
        metavar="PIXELS",
        help=(
            "full-resolution geometry pixel guard (default: "
            f"{DEFAULT_MAX_GEOMETRY_PIXELS}; use 'none' to disable)"
        ),
    )
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--no-support-removal", action="store_true")
    parser.add_argument("--no-tight-crop", action="store_true")
    args = parser.parse_args()
    try:
        max_geometry_pixels = parse_max_geometry_pixels(args.max_geometry_pixels)
    except ValueError as exc:
        parser.error(str(exc))
    config = GeometryConfig(
        max_deskew_degrees=0.0 if args.no_deskew else 45.0,
        max_geometry_pixels=max_geometry_pixels,
    )
    result = process(
        args.input,
        args.output,
        config,
        confidence_floor=args.confidence_floor,
        remove_support=not args.no_support_removal,
        deskew=not args.no_deskew,
        tight_crop_enabled=not args.no_tight_crop,
    )
    print(json.dumps(_result_json(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
