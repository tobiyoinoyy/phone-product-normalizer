"""Installable API for phone product image normalization.

The high-level package defaults to ``ZhengPeng7/BiRefNet_dynamic`` and keeps
the older ``scripts/normalize_ben2.py`` command available for compatibility.
Heavy model libraries are loaded only when a backend is instantiated.
"""

from .api import (
    DEFAULT_MAX_GEOMETRY_PIXELS,
    DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE,
    DEFAULT_BIREFNET_MAX_SIDE,
    DEFAULT_BACKEND,
    DEFAULT_MODEL,
    IMAGE_EXTENSIONS,
    BatchResult,
    FileResult,
    Normalizer,
    NormalizerConfig,
    ProcessingInfo,
    available_backends,
    collect_image_paths,
    create_segmentation_backend,
    process_many,
    parse_max_geometry_pixels,
    register_backend,
    validate_geometry_size,
)

__all__ = [
    "DEFAULT_BIREFNET_MAX_SIDE",
    "DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE",
    "DEFAULT_MAX_GEOMETRY_PIXELS",
    "DEFAULT_BACKEND",
    "DEFAULT_MODEL",
    "IMAGE_EXTENSIONS",
    "BatchResult",
    "FileResult",
    "Normalizer",
    "NormalizerConfig",
    "ProcessingInfo",
    "available_backends",
    "collect_image_paths",
    "create_segmentation_backend",
    "process_many",
    "parse_max_geometry_pixels",
    "register_backend",
    "validate_geometry_size",
]

__version__ = "0.1.0"
