"""BiRefNet_dynamic phone product image normalization API."""

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
    collect_image_paths,
    create_segmentation_backend,
    process_many,
    parse_max_geometry_pixels,
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
    "collect_image_paths",
    "create_segmentation_backend",
    "process_many",
    "parse_max_geometry_pixels",
    "validate_geometry_size",
]

__version__ = "0.3.0"
