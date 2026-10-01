"""Court detection & calibration (Phase 1)."""

from . import geometry
from .calibration import CourtCalibration, require_calibration
from .homography import (
    CalibrationError,
    VerificationReport,
    apply_homography,
    court_to_pixel,
    estimate_homography,
    pixel_to_court,
    verify_calibration,
)
from .line_detector import DetectionResult, DetectorConfig, detect_court

__all__ = [
    "geometry",
    "CourtCalibration",
    "require_calibration",
    "CalibrationError",
    "VerificationReport",
    "apply_homography",
    "court_to_pixel",
    "pixel_to_court",
    "estimate_homography",
    "verify_calibration",
    "DetectionResult",
    "DetectorConfig",
    "detect_court",
]
