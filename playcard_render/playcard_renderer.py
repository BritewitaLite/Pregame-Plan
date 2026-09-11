"""
playcard_renderer.py
---------------------
Renders a per-frame football field play card that shows every tracked
player's position projected onto a canonical top-down field diagram.

The homography (H) computed by HoughFieldHomography maps image pixels
→ bird's-eye-view pixels.  We use it here to project the foot-point of
each player's bounding box onto the play card.

Output modes
------------
A) Per-frame play card image  (cv2 ndarray, BEV_W × BEV_H px)
B) Side-by-side composite     (original frame | play card)
C) Embedded mini-map          (play card inset in corner of the frame)
D) Video of play cards        (saved as .avi)

Usage
-----
    from playcard_renderer import PlaycardRenderer

    renderer = PlaycardRenderer(team_a_color=(50, 100, 220),   # BGR blue  – offense
                                team_b_color=(220, 50, 50),    # BGR red   – defense
                                ball_color=(255, 200, 0))

    # Generate one composite frame
    composite = renderer.render_composite(video_frame, H, frame_tracks)

    # Or generate a full video
    composites = renderer.render_video(video_frames, homographies, tracks)
    from utils import save_video
    save_video(composites, 'output_videos/playcard_video.avi')
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple, Any

# ── Field geometry (must match hough_field_homography.py) ──────────────────
FIELD_W_YD    = 53.3
FIELD_L_YD    = 120.0
PIXELS_PER_YD = 10
BEV_W  = int(FIELD_W_YD  * PIXELS_PER_YD)   # 533
BEV_H  = int(FIELD_L_YD  * PIXELS_PER_YD)   # 1200

# Yard-line labels (every 10 yards, both directions from 50)
YARD_LABELS = {
    10:  "10", 20: "20", 30: "30", 40: "40", 50: "50",
    60:  "40", 70: "30", 80: "20", 90: "10",
}

# ── Design palette ──────────────────────────────────────────────────────────
TURF_DARK   = (34,  85,  34)   # dark green stripe   (BGR)
TURF_LIGHT  = (42, 100,  42)   # light green stripe
LINE_COLOR  = (255, 255, 255)  # white yard lines
ENDZONE_A   = (20,  60, 140)   # deep navy  (near end zone)
ENDZONE_B   = (140, 20,  20)   # deep crimson (far end zone)
HASH_COLOR  = (200, 200, 200)  # hash marks
SHADOW_ALPHA = 0.35


class PlaycardRenderer:
    """
    Renders tracked players onto a top-down football field play card
    using a Hough-derived homography matrix.
    """

    def __init__(
        self,
        team_a_color: Tuple[int,int,int] = (50, 100, 220),    # offense BGR (blue)
        team_b_color: Tuple[int,int,int] = (220, 50, 50),     # defense BGR (red)
        ball_color:   Tuple[int,int,int] = (  0, 200, 255),
        player_radius: int = 9,
        trail_length:  int = 20,        # frames of movement trail to draw
        minimap_scale: float = 0.28,    # fraction of frame height for inset
    ):
        self.team_a_color  = team_a_color
        self.team_b_color  = team_b_color
        self.ball_color    = ball_color
        self.player_radius = player_radius
        self.trail_length  = trail_length
        self.minimap_scale = minimap_scale

        # Trail history: {track_id: [(bev_x, bev_y), ...]}
        self._trails: Dict[int, List[Tuple[float, float]]] = {}

        # Pre-render the static field background once
        self._field_bg: Optional[np.ndarray] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def render_video(
        self,
        video_frames: List[np.ndarray],
        homographies: List[Optional[np.ndarray]],
        tracks: Dict[str, List[Dict]],
        mode: str = "composite",      # "composite" | "minimap" | "playcard"
    ) -> List[np.ndarray]:
        """
        Process all frames.  Returns a list of rendered images.
        mode:
          'composite'  – original frame (left) + play card (right), same height
          'minimap'    – original frame with play card inset in bottom-right
          'playcard'   – play card only
        """
        self._trails.clear()
        out = []
        n = len(video_frames)

        players_all = tracks.get('players',   [{}] * n)
        refs_all    = tracks.get('referees',  [{}] * n)
        ball_all    = tracks.get('ball',      [{}] * n)

        for i, (frame, H) in enumerate(zip(video_frames, homographies)):
            frame_players = players_all[i] if i < len(players_all) else {}
            frame_refs    = refs_all[i]    if i < len(refs_all)    else {}
            frame_ball    = ball_all[i]    if i < len(ball_all)    else {}

            frame_tracks = {
                'players':  frame_players,
                'referees': frame_refs,
                'ball':     frame_ball,
            }

            if mode == "composite":
                rendered = self.render_composite(frame, H, frame_tracks)
            elif mode == "minimap":
                rendered = self.render_minimap(frame, H, frame_tracks)
            else:
                rendered = self.render_playcard(H, frame_tracks)
                # playcard alone — resize to match frame height
                fh = frame.shape[0]
                fw = int(BEV_W * fh / BEV_H)
                rendered = cv2.resize(rendered, (fw, fh))

            out.append(rendered)
        return out

    def render_playcard(
        self,
        H: Optional[np.ndarray],
        frame_tracks: Dict[str, Any],
    ) -> np.ndarray:
        """Return a BEV_W × BEV_H play card image for one frame."""
        card = self._get_field_bg().copy()

        if H is None:
            self._draw_no_signal(card)
            return card

        self._draw_players(card, H, frame_tracks)
        self._draw_frame_label(card)
        return card

    def render_composite(
        self,
        frame: np.ndarray,
        H: Optional[np.ndarray],
        frame_tracks: Dict[str, Any],
    ) -> np.ndarray:
        """
        Returns a side-by-side image: original frame (left) | play card (right).
        Both panels share the same height.
        """
        fh, fw = frame.shape[:2]
        card   = self.render_playcard(H, frame_tracks)

        # Scale play card to match frame height
        card_w = int(BEV_W * fh / BEV_H)
        card_scaled = cv2.resize(card, (card_w, fh), interpolation=cv2.INTER_AREA)

        # Divider line
        divider = np.full((fh, 4, 3), 40, dtype=np.uint8)
        composite = np.concatenate([frame, divider, card_scaled], axis=1)
        return composite

    def render_minimap(
        self,
        frame: np.ndarray,
        H: Optional[np.ndarray],
        frame_tracks: Dict[str, Any],
    ) -> np.ndarray:
        """
        Returns the original frame with the play card inset in the
        bottom-right corner (semi-transparent border, rounded feel).
        """
        out   = frame.copy()
        fh, fw = frame.shape[:2]

        mm_h  = int(fh * self.minimap_scale)
        mm_w  = int(BEV_W * mm_h / BEV_H)

        card  = self.render_playcard(H, frame_tracks)
        card_small = cv2.resize(card, (mm_w, mm_h), interpolation=cv2.INTER_AREA)

        # Drop-shadow rectangle
        pad   = 6
        x0    = fw - mm_w - pad * 2
        y0    = fh - mm_h - pad * 2

        shadow = out[y0:y0+mm_h+pad*2, x0:x0+mm_w+pad*2].copy()
        shadow[:] = (15, 15, 15)
        cv2.addWeighted(shadow, 0.55,
                        out[y0:y0+mm_h+pad*2, x0:x0+mm_w+pad*2], 0.45,
                        0, out[y0:y0+mm_h+pad*2, x0:x0+mm_w+pad*2])

        # Paste play card
        x1, y1 = x0 + pad, y0 + pad
        out[y1:y1+mm_h, x1:x1+mm_w] = card_small

        # White border
        cv2.rectangle(out, (x1-1, y1-1), (x1+mm_w, y1+mm_h), (255,255,255), 1)

        # Label
        cv2.putText(out, "PLAY CARD", (x1+4, y1+14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)
        return out

    # ── Field background ────────────────────────────────────────────────────

    def _get_field_bg(self) -> np.ndarray:
        if self._field_bg is not None:
            return self._field_bg.copy()
        self._field_bg = self._draw_field()
        return self._field_bg.copy()

    def _draw_field(self) -> np.ndarray:
        """Render a clean top-down football field diagram."""
        img = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

        # ── Alternating 5-yard turf stripes ────────────────────────────────
        stripe_yd = 5
        stripe_px = stripe_yd * PIXELS_PER_YD
        for i in range(int(FIELD_L_YD / stripe_yd)):
            color = TURF_LIGHT if i % 2 == 0 else TURF_DARK
            y0 = i * stripe_px
            y1 = y0 + stripe_px
            img[y0:y1, :] = color

        # ── End zones ──────────────────────────────────────────────────────
        ez_px = int(10 * PIXELS_PER_YD)   # 10-yard end zones
        img[:ez_px,      :] = ENDZONE_A
        img[BEV_H-ez_px:, :] = ENDZONE_B

        # ── Yard lines every 5 yards ────────────────────────────────────────
        for y_yd in range(0, int(FIELD_L_YD) + 1, 5):
            y_px = int(y_yd * PIXELS_PER_YD)
            thickness = 2 if y_yd % 10 == 0 else 1
            cv2.line(img, (0, y_px), (BEV_W, y_px), LINE_COLOR, thickness)

        # ── Sidelines ──────────────────────────────────────────────────────
        cv2.line(img, (0, 0), (0, BEV_H), LINE_COLOR, 3)
        cv2.line(img, (BEV_W-1, 0), (BEV_W-1, BEV_H), LINE_COLOR, 3)

        # ── Hash marks (every yard between the 10s) ─────────────────────────
        hash_inset_px = int(18.5 * PIXELS_PER_YD / 3)  # ~60 ft from each sideline (college)
        hash_len      = 4
        for y_yd in range(10, int(FIELD_L_YD) - 10, 1):
            y_px = int(y_yd * PIXELS_PER_YD)
            # Left hash
            cv2.line(img, (hash_inset_px, y_px),
                          (hash_inset_px + hash_len, y_px), HASH_COLOR, 1)
            # Right hash
            cv2.line(img, (BEV_W - hash_inset_px - hash_len, y_px),
                          (BEV_W - hash_inset_px, y_px), HASH_COLOR, 1)

        # ── Yard number labels ──────────────────────────────────────────────
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        for y_yd, label in YARD_LABELS.items():
            y_px = int(y_yd * PIXELS_PER_YD)
            # Left side (rotated via transpose trick)
            self._draw_rotated_text(img, label, (18, y_px), font, font_scale,
                                    (220, 220, 220), angle=90)
            # Right side
            self._draw_rotated_text(img, label, (BEV_W - 18, y_px), font, font_scale,
                                    (220, 220, 220), angle=90)

        # ── Goal posts (schematic) ──────────────────────────────────────────
        mid = BEV_W // 2
        for y_px in [ez_px // 2, BEV_H - ez_px // 2]:
            cv2.line(img, (mid, y_px - 12), (mid, y_px + 4), (200, 200, 100), 2)
            cv2.line(img, (mid - 10, y_px - 12), (mid + 10, y_px - 12),
                     (200, 200, 100), 2)

        return img

    def _draw_rotated_text(
        self, img, text, center, font, scale, color, thickness=1, angle=0
    ):
        """Draw text rotated by `angle` degrees around `center`."""
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tmp = np.zeros((th + 8, tw + 8, 3), dtype=np.uint8)
        cv2.putText(tmp, text, (4, th + 2), font, scale, color, thickness, cv2.LINE_AA)
        M   = cv2.getRotationMatrix2D((tw//2 + 4, th//2 + 4), angle, 1)
        rot = cv2.warpAffine(tmp, M, (tw + 8, th + 8))
        cx, cy = center
        x0, y0 = cx - tw//2 - 4, cy - th//2 - 4
        x1, y1 = x0 + rot.shape[1], y0 + rot.shape[0]
        # clip to image bounds
        ix0 = max(0, x0); iy0 = max(0, y0)
        ix1 = min(img.shape[1], x1); iy1 = min(img.shape[0], y1)
        rx0 = ix0 - x0; ry0 = iy0 - y0
        rx1 = rx0 + (ix1 - ix0); ry1 = ry0 + (iy1 - iy0)
        if ix1 > ix0 and iy1 > iy0:
            patch = rot[ry0:ry1, rx0:rx1]
            mask  = (patch.sum(axis=2) > 0)
            img[iy0:iy1, ix0:ix1][mask] = patch[mask]

    # ── Player projection ───────────────────────────────────────────────────

    def _project_foot(
        self,
        bbox: List[float],
        H: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """
        Project the foot point (centre-bottom) of a bounding box
        through H into BEV pixel coordinates.
        """
        foot_x = (bbox[0] + bbox[2]) / 2.0
        foot_y = float(bbox[3])
        pt = np.array([[[foot_x, foot_y]]], dtype=np.float64)
        bev = cv2.perspectiveTransform(pt, H)[0][0]
        bx, by = int(round(bev[0])), int(round(bev[1]))
        # Accept points slightly outside (camera may see partial players)
        if -20 <= bx <= BEV_W + 20 and -20 <= by <= BEV_H + 20:
            return bx, by
        return None

    def _draw_players(
        self,
        card: np.ndarray,
        H: np.ndarray,
        frame_tracks: Dict[str, Any],
    ):
        """Project all tracked entities onto the play card."""
        r = self.player_radius

        # ── Referees (grey, small, no trail) ───────────────────────────────
        for tid, info in frame_tracks.get('referees', {}).items():
            pt = self._project_foot(info['bbox'], H)
            if pt:
                cv2.circle(card, pt, r - 3, (160, 160, 160), -1)
                cv2.circle(card, pt, r - 3, (220, 220, 220),  1)

        # ── Players ─────────────────────────────────────────────────────────
        # We split into two teams by track_id parity as a reasonable heuristic
        # (replace with jersey-colour logic when available)
        for tid, info in frame_tracks.get('players', {}).items():
            pt = self._project_foot(info['bbox'], H)
            if pt is None:
                continue

            # Update trail
            trail = self._trails.setdefault(tid, [])
            trail.append(pt)
            if len(trail) > self.trail_length:
                trail.pop(0)

            # Team colour by id parity — swap to real team assignment when ready
            color = self.team_a_color if tid % 2 == 0 else self.team_b_color

            # Draw movement trail (fading)
            if len(trail) > 1:
                for j in range(1, len(trail)):
                    alpha = j / len(trail)
                    tc    = tuple(int(c * alpha) for c in color)
                    cv2.line(card, trail[j-1], trail[j], tc, 1)

            # Drop shadow
            shadow_pt = (pt[0] + 2, pt[1] + 2)
            cv2.circle(card, shadow_pt, r + 1, (10, 10, 10), -1)

            # Player dot — filled with team colour
            cv2.circle(card, pt, r,     color,   -1)
            cv2.circle(card, pt, r,     (255,255,255), 1)   # white border

            # Highlight cap
            cap = (pt[0] - 2, pt[1] - 2)
            cv2.circle(card, cap, r // 3,
                       tuple(min(255, c + 80) for c in color), -1)

            # Track-ID label
            label = str(tid)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.32, 1)
            cv2.putText(card, label,
                        (pt[0] - tw // 2, pt[1] + th // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.32, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # ── Ball ────────────────────────────────────────────────────────────
        for tid, info in frame_tracks.get('ball', {}).items():
            pt = self._project_foot(info['bbox'], H)
            if pt:
                cv2.circle(card, pt, 5, self.ball_color,  -1)
                cv2.circle(card, pt, 5, (255, 255, 255),   1)
                cv2.circle(card, pt, 2, (255, 255, 255),  -1)

    def _draw_frame_label(self, card: np.ndarray):
        """Minimal watermark so coaches know the source."""
        cv2.putText(card, "PREGAME PLAN  |  HOUGH HOMOGRAPHY",
                    (6, BEV_H - 6), cv2.FONT_HERSHEY_DUPLEX,
                    0.3, (150, 150, 150), 1, cv2.LINE_AA)

    def _draw_no_signal(self, card: np.ndarray):
        cv2.putText(card, "NO HOMOGRAPHY", (BEV_W // 4, BEV_H // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 220), 2, cv2.LINE_AA)


# ── Snapshot helper ─────────────────────────────────────────────────────────

def save_playcard_snapshot(
    frame: np.ndarray,
    H: Optional[np.ndarray],
    frame_tracks: Dict[str, Any],
    out_path: str = "output_videos/playcard_snapshot.jpg",
    mode: str = "composite",
):
    """
    Save a single-frame play card image to disk.
    Useful for quick debugging or exporting a still for a PDF scout card.
    """
    renderer  = PlaycardRenderer()
    if mode == "minimap":
        img = renderer.render_minimap(frame, H, frame_tracks)
    elif mode == "playcard":
        img = renderer.render_playcard(H, frame_tracks)
    else:
        img = renderer.render_composite(frame, H, frame_tracks)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"Saved play card snapshot → {out_path}")
    return img


import os
