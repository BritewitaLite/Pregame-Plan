"""
hough_field_homography.py
--------------------------
Computes a football field homography (image → top-down bird's-eye view)
using ONLY the Hough Line Transform — no keypoint detection model required.

Pipeline:
  1. Pre-process frame  →  edge / colour mask focused on field lines
  2. Probabilistic Hough  →  raw line segments
  3. Cluster lines into two dominant orientations (sidelines vs yard lines)
  4. Intersect representative lines from each group  →  4 corner candidates
  5. RANSAC-style refinement across frames  →  stable homography H
  6. Warp frame  →  bird's-eye field image; map any (x,y) pixel to field coords

Usage (standalone):
    detector = HoughFieldHomography()
    result   = detector.process_frame(frame)        # single frame
    H        = detector.H                           # 3×3 homography matrix
    pt_field = detector.image_to_field(px, py)      # map pixel → field yards
    bev      = detector.warp_to_bev(frame)          # bird's-eye-view image

Drop-in replacement for FieldKeypointDetector in main.py:
    from hough_field_homography import HoughFieldHomography
    field_mapper = HoughFieldHomography()
    field_mapper.process_video(video_frames)         # fit on all frames
"""

import cv2
import numpy as np
import pickle
import os
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Real-world football field dimensions (yards, origin = back of one end zone)
# ---------------------------------------------------------------------------
FIELD_W_YD   = 53.3          # width  (sideline to sideline)
FIELD_L_YD   = 120.0         # length (end zone to end zone)
PIXELS_PER_YD = 10           # resolution of the bird's-eye-view output image
BEV_W  = int(FIELD_W_YD  * PIXELS_PER_YD)   # 533 px
BEV_H  = int(FIELD_L_YD  * PIXELS_PER_YD)   # 1200 px

# Corners of the field in BEV pixel space  (TL, TR, BR, BL)
BEV_CORNERS = np.array([
    [0,       0      ],
    [BEV_W-1, 0      ],
    [BEV_W-1, BEV_H-1],
    [0,       BEV_H-1],
], dtype=np.float32)


class HoughFieldHomography:
    """
    Estimates a projective homography from broadcast/film camera view to a
    canonical top-down football field using Hough line detection only.
    """

    def __init__(
        self,
        rho: float          = 1,
        theta: float        = np.pi / 180,
        hough_thresh: int   = 50,  # Reduced from 80
        min_line_len: int   = 50,  # Reduced from 80
        max_line_gap: int   = 20,  # Reduced from 30
        angle_tol_deg: float = 15.0,
        n_angle_bins: int   = 180,
        green_thresh: int   = 40,       # HSV green saturation floor
        debug: bool         = False,
    ):
        self.rho           = rho
        self.theta         = theta
        self.hough_thresh  = hough_thresh
        self.min_line_len  = min_line_len
        self.max_line_gap  = max_line_gap
        self.angle_tol     = np.deg2rad(angle_tol_deg)
        self.n_angle_bins  = n_angle_bins
        self.green_thresh  = green_thresh
        self.debug         = debug

        self.H: Optional[np.ndarray] = None          # current homography
        self._H_history: List[np.ndarray] = []       # per-frame history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str = "stubs/hough_homography.pkl",
    ) -> List[Optional[np.ndarray]]:
        """
        Compute per-frame homographies.  Returns list of H matrices
        (None for frames where detection failed).
        Mirrors the stub-caching pattern used by FieldKeypointDetector.
        """
        if read_from_stub and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        results = []
        prev_H  = None
        for frame in frames:
            H = self.process_frame(frame, fallback_H=prev_H)
            results.append(H.copy() if H is not None else None)
            if H is not None:
                prev_H = H
        self._H_history = [h for h in results if h is not None]

        os.makedirs(os.path.dirname(stub_path) if os.path.dirname(stub_path) else ".", exist_ok=True)
        with open(stub_path, "wb") as f:
            pickle.dump(results, f)
        return results

    def process_frame(
        self,
        frame: np.ndarray,
        fallback_H: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Detect field lines in one frame and compute / update self.H.
        Returns the homography matrix or None on failure.
        """
        mask   = self._field_mask(frame)
        edges  = self._edge_map(frame, mask)
        lines  = self._hough_lines(edges)

        if lines is None or len(lines) < 4:
            self.H = fallback_H
            return self.H

        horiz, vert = self._cluster_lines(lines, frame.shape)

        if len(horiz) < 2 or len(vert) < 2:
            self.H = fallback_H
            return self.H

        corners = self._find_corners(horiz, vert, frame.shape)

        if corners is None:
            self.H = fallback_H
            return self.H

        H, _ = cv2.findHomography(corners, BEV_CORNERS, cv2.RANSAC, 5.0)
        if H is not None:
            # Assign H, then verify orientation: sometimes the BEV axes
            # get swapped (x/y) which lays players across the length.
            # Detect that case by mapping small horizontal/vertical
            # offsets in image space into BEV and comparing resulting
            # distances in yards. If the mapped horizontal span
            # corresponds to the field length rather than width, swap
            # the BEV axes by pre-multiplying with a permutation matrix.
            self.H = H
            try:
                h, w = frame.shape[:2]
                cx, cy = w / 2.0, h / 2.0
                dx = max(10, w * 0.1)
                dy = max(10, h * 0.1)
                pts = np.array([[[cx - dx, cy]], [[cx + dx, cy]], [[cx, cy - dy]], [[cx, cy + dy]]], dtype=np.float64)
                bev_pts = cv2.perspectiveTransform(pts, self.H)
                left_bev  = bev_pts[0][0]
                right_bev = bev_pts[1][0]
                top_bev   = bev_pts[2][0]
                bot_bev   = bev_pts[3][0]
                width_yd  = abs(right_bev[0] - left_bev[0]) / PIXELS_PER_YD
                height_yd = abs(bot_bev[1] - top_bev[1]) / PIXELS_PER_YD
                # If the horizontal image displacement maps to a distance
                # larger than the real field width, axes are likely swapped.
                if width_yd > FIELD_W_YD * 0.75 and height_yd < FIELD_W_YD * 0.75:
                    S = np.array([[0.0, 1.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0]], dtype=np.float64)
                    self.H = S @ self.H
            except Exception:
                # Non-fatal: if detection fails, keep original H
                pass
        else:
            self.H = fallback_H

        return self.H

    def warp_to_bev(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return a bird's-eye-view warp of the frame, or None if no H."""
        if self.H is None:
            return None
        return cv2.warpPerspective(frame, self.H, (BEV_W, BEV_H))

    def image_to_field(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """
        Map an image pixel (px, py) → field coordinates in yards.
        Returns (x_yd, y_yd) with origin at back of near end zone,
        x across the width, y down the length.
        """
        if self.H is None:
            return None
        pt = np.array([[[px, py]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(pt, self.H)[0][0]
        x_yd = dst[0] / PIXELS_PER_YD
        y_yd = dst[1] / PIXELS_PER_YD
        return float(x_yd), float(y_yd)

    def draw_annotations(
        self,
        frames: List[np.ndarray],
        homographies: List[Optional[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Overlay detected field lines and grid on each frame.
        Mirrors the annotation pattern used by the keypoint detector.
        """
        out = []
        for frame, H in zip(frames, homographies):
            annotated = frame.copy()
            # Re-run Hough on this frame just for visualisation
            mask  = self._field_mask(frame)
            edges = self._edge_map(frame, mask)
            lines = self._hough_lines(edges)
            if lines is not None:
                for seg in lines:
                    x1, y1, x2, y2 = seg[0]
                    cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 255), 1)
            # Draw homography status
            status = "H OK" if H is not None else "H FAIL"
            color  = (0, 255, 0) if H is not None else (0, 0, 255)
            cv2.putText(annotated, f"Hough Homography: {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            out.append(annotated)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _field_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Binary mask that keeps only pixels likely to be the green playing
        surface, suppressing crowd / sky / scoreboards.
        Tuned for typical broadcast football footage.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Broad green range (handles shadows and different turf colours)
        lower1 = np.array([30,  self.green_thresh, 40])
        upper1 = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower1, upper1)

        # White line pixels (high value, low saturation)
        lower_w = np.array([0, 0, 200])
        upper_w = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_w, upper_w)

        # Keep white only where it neighbours green (field lines only)
        kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        green_dilated = cv2.dilate(green_mask, kernel, iterations=2)
        line_mask  = cv2.bitwise_and(white_mask, green_dilated)

        combined   = cv2.bitwise_or(green_mask, line_mask)

        # Morphological clean-up
        combined   = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        combined   = cv2.morphologyEx(combined, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return combined

    def _edge_map(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Canny edges restricted to the field region.
        """
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        blur   = cv2.GaussianBlur(masked, (5, 5), 0)
        edges  = cv2.Canny(blur, 30, 90)
        return edges

    def _hough_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """Probabilistic Hough line transform."""
        lines = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.hough_thresh,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap,
        )
        return lines  # shape (N, 1, 4) or None

    def _line_angle(self, x1, y1, x2, y2) -> float:
        """Angle in [-pi/2, pi/2)."""
        return np.arctan2(y2 - y1, x2 - x1) % np.pi - (np.pi / 2 if
               np.arctan2(y2 - y1, x2 - x1) % np.pi >= np.pi / 2 else 0)

    def _cluster_lines(
        self,
        lines: np.ndarray,
        shape: Tuple[int, ...],
    ) -> Tuple[List, List]:
        """
        Split lines into two dominant orientation groups using angle histogram.

        Football fields have two families:
          • Horizontal-ish  — sidelines and end lines
          • Vertical-ish    — yard lines (hash marks and field stripes)

        The two peaks of the angle histogram identify these groups.
        """
        h, w = shape[:2]
        angles = []
        segs   = lines[:, 0, :]     # (N, 4)

        for x1, y1, x2, y2 in segs:
            a = np.arctan2(float(y2 - y1), float(x2 - x1))
            a = a % np.pi           # map to [0, π)
            angles.append(a)

        angles = np.array(angles)

        # Histogram with 180 bins (1 degree resolution)
        hist, bin_edges = np.histogram(angles, bins=self.n_angle_bins, range=(0, np.pi))

        # Smooth histogram and find two dominant peaks
        hist_smooth = np.convolve(hist, np.ones(5) / 5, mode='same')
        peaks       = self._two_dominant_peaks(hist_smooth)

        if len(peaks) < 2:
            # Fallback: assume near-horizontal and near-vertical
            peaks = [0, self.n_angle_bins // 2]

        peak_angles = [bin_edges[p] + (np.pi / self.n_angle_bins) for p in peaks]
        # Sort so peak_angles[0] is the more horizontal one (closer to 0 or π)
        peak_angles.sort(key=lambda a: min(abs(a), abs(a - np.pi)))

        horiz_center = peak_angles[0]
        vert_center  = peak_angles[1] if len(peak_angles) > 1 else horiz_center + np.pi / 2

        horiz_segs, vert_segs = [], []
        for i, (x1, y1, x2, y2) in enumerate(segs):
            a = angles[i]
            d_h = min(abs(a - horiz_center), np.pi - abs(a - horiz_center))
            d_v = min(abs(a - vert_center),  np.pi - abs(a - vert_center))
            length = np.hypot(x2 - x1, y2 - y1)
            if d_h < self.angle_tol and d_h <= d_v:
                horiz_segs.append((x1, y1, x2, y2, length))
            elif d_v < self.angle_tol:
                vert_segs.append((x1, y1, x2, y2, length))

        # Keep the longest lines as most representative
        horiz_segs.sort(key=lambda s: -s[4])
        vert_segs.sort(key=lambda s:  -s[4])
        return horiz_segs, vert_segs

    def _two_dominant_peaks(self, hist: np.ndarray) -> List[int]:
        """Return indices of the two highest non-adjacent peaks in histogram."""
        sorted_idx  = np.argsort(hist)[::-1]
        min_sep     = len(hist) // 6      # peaks must be at least 30° apart
        peaks       = []
        for idx in sorted_idx:
            if all(abs(idx - p) >= min_sep for p in peaks):
                peaks.append(int(idx))
            if len(peaks) == 2:
                break
        return peaks

    def _line_to_params(self, x1, y1, x2, y2):
        """Return (a, b, c) of line ax + by + c = 0, normalised."""
        dx, dy = x2 - x1, y2 - y1
        a, b, c = dy, -dx, dx * y1 - dy * x1
        n = np.hypot(a, b) + 1e-9
        return a / n, b / n, c / n

    def _intersect(self, l1, l2) -> Optional[Tuple[float, float]]:
        """Intersection of two lines given as (a,b,c)."""
        a1, b1, c1 = l1
        a2, b2, c2 = l2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        return x, y

    def _find_corners(
        self,
        horiz_segs: List,
        vert_segs:  List,
        shape: Tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """
        Find four corners of the visible field quadrilateral by intersecting
        the two most extreme horizontal lines with the two most extreme
        vertical lines (i.e., outermost sidelines and yard-line boundaries).
        """
        h, w = shape[:2]

        def seg_to_extended(x1, y1, x2, y2, W, H):
            """Extend a segment to the full image width or height."""
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 1e-3:
                return (x1, 0, x1, H)
            if abs(dy) < 1e-3:
                return (0, y1, W, y1)
            # Parametric: find t at x=0 and x=W
            t0 = -x1 / dx
            t1 = (W - x1) / dx
            return (0, int(y1 + t0 * dy), W, int(y1 + t1 * dy))

        # Pick the two most extreme horizontal lines (top & bottom of visible field)
        # "extreme" = sorted by their median y coordinate
        def median_y(seg):
            x1, y1, x2, y2, *_ = seg
            return (y1 + y2) / 2.0

        def median_x(seg):
            x1, y1, x2, y2, *_ = seg
            return (x1 + x2) / 2.0

        horiz_by_y = sorted(horiz_segs, key=median_y)
        vert_by_x  = sorted(vert_segs,  key=median_x)

        top_seg    = horiz_by_y[0][:4]
        bot_seg    = horiz_by_y[-1][:4]
        left_seg   = vert_by_x[0][:4]
        right_seg  = vert_by_x[-1][:4]

        # Convert to line params
        top_line   = self._line_to_params(*top_seg)
        bot_line   = self._line_to_params(*bot_seg)
        left_line  = self._line_to_params(*left_seg)
        right_line = self._line_to_params(*right_seg)

        tl = self._intersect(top_line, left_line)
        tr = self._intersect(top_line, right_line)
        br = self._intersect(bot_line, right_line)
        bl = self._intersect(bot_line, left_line)

        if any(pt is None for pt in [tl, tr, br, bl]):
            return None

        corners = np.array([tl, tr, br, bl], dtype=np.float32)

        # Sanity check: corners should be within a reasonable margin of the image
        margin = max(w, h) * 0.5
        for cx, cy in corners:
            if not (-margin <= cx <= w + margin and -margin <= cy <= h + margin):
                return None

        return corners


# ---------------------------------------------------------------------------
# Annotator  (mirrors FieldKeypointAnnotator interface)
# ---------------------------------------------------------------------------

class HoughFieldAnnotator:
    """Draw Hough-derived field lines and homography grid overlay."""

    def draw(
        self,
        frames: List[np.ndarray],
        homographies: List[Optional[np.ndarray]],
        draw_grid: bool = True,
    ) -> List[np.ndarray]:
        out = []
        for frame, H in zip(frames, homographies):
            out.append(self._annotate(frame, H, draw_grid))
        return out

    def _annotate(
        self,
        frame: np.ndarray,
        H: Optional[np.ndarray],
        draw_grid: bool,
    ) -> np.ndarray:
        ann = frame.copy()
        if H is None:
            cv2.putText(ann, "No Homography", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return ann

        H_inv = np.linalg.inv(H)

        if draw_grid:
            # Project yard lines every 10 yards back onto the image
            for y_yd in range(0, int(FIELD_L_YD) + 1, 10):
                y_px = y_yd * PIXELS_PER_YD
                pts_bev = np.array([[[0, y_px], [BEV_W, y_px]]], dtype=np.float64)
                pts_img = cv2.perspectiveTransform(pts_bev, H_inv)
                p1 = tuple(pts_img[0][0].astype(int))
                p2 = tuple(pts_img[0][1].astype(int))
                cv2.line(ann, p1, p2, (255, 200, 0), 1)

            # Project sidelines
            for x_yd in [0, int(FIELD_W_YD)]:
                x_px = x_yd * PIXELS_PER_YD
                pts_bev = np.array([[[x_px, 0], [x_px, BEV_H]]], dtype=np.float64)
                pts_img = cv2.perspectiveTransform(pts_bev, H_inv)
                p1 = tuple(pts_img[0][0].astype(int))
                p2 = tuple(pts_img[0][1].astype(int))
                cv2.line(ann, p1, p2, (0, 200, 255), 2)

        cv2.putText(ann, "Hough Homography", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return ann


# ---------------------------------------------------------------------------
# Integration helper  — drop-in swap for main.py
# ---------------------------------------------------------------------------

def build_hough_mapper(
    video_frames: List[np.ndarray],
    read_from_stub: bool = False,
    stub_path: str = "stubs/hough_homography.pkl",
) -> Tuple["HoughFieldHomography", List[Optional[np.ndarray]]]:
    """
    Convenience factory.  Returns (mapper, per_frame_homographies).

    In main.py replace:
        field_keypoint_detector  = FieldKeypointDetector('models/...')
        field_keypoints          = field_keypoint_detector.get_field_keypoints(...)
    With:
        from hough_field_homography import build_hough_mapper, HoughFieldAnnotator
        mapper, homographies = build_hough_mapper(video_frames)
        annotator            = HoughFieldAnnotator()
        output_frames        = annotator.draw(output_frames, homographies)
    """
    mapper       = HoughFieldHomography()
    homographies = mapper.process_video(
        video_frames,
        read_from_stub=read_from_stub,
        stub_path=stub_path,
    )
    return mapper, homographies
