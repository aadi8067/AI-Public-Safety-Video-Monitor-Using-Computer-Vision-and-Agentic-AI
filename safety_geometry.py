"""Geometry and trajectory helpers shared by safety detectors."""
import math
import cv2
import numpy as np


def normalize_polygon(polygon):
    """Return a safe list of [x, y] float points from nested/flat inputs.

    Accepts Python lists/tuples and numpy arrays shaped (N,2), (1,N,2),
    or a flat [x1,y1,x2,y2,...]. Invalid input returns [].
    """
    if polygon is None:
        return []
    try:
        arr = np.asarray(polygon)
        if arr.size == 0:
            return []
        if arr.ndim == 1:
            if arr.size % 2 != 0:
                return []
            arr = arr.reshape(-1, 2)
        else:
            arr = arr.reshape(-1, 2)
        out = []
        for pt in arr:
            out.append([float(pt[0]), float(pt[1])])
        return out if len(out) >= 3 else []
    except (TypeError, ValueError):
        return []


def center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def foot_point(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, y2)


def width_height(bbox):
    return max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])


def aspect_ratio(bbox):
    w, h = width_height(bbox)
    return w / h


def distance(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def point_in_polygon(point, polygon):
    if not polygon:
        return False
    contour = np.asarray(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def polygon_mask_area(polygon, width, height):
    if not polygon:
        return float(width * height)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return float(cv2.countNonZero(mask))


def bbox_area(bbox):
    w, h = width_height(bbox)
    return w * h


def normalized_density(person_bboxes, roi_polygon, width, height):
    roi_area = polygon_mask_area(roi_polygon, width, height)
    if roi_area <= 0:
        return 0.0
    occupied = sum(bbox_area(b) for b in person_bboxes)
    return min(1.0, occupied / roi_area)


def angle_between(v1, v2):
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def clamp_bbox(bbox, width, height):
    return [
        max(0.0, min(float(width - 1), float(bbox[0]))),
        max(0.0, min(float(height - 1), float(bbox[1]))),
        max(0.0, min(float(width - 1), float(bbox[2]))),
        max(0.0, min(float(height - 1), float(bbox[3]))),
    ]
