"""Locate the row band occupied by subtitles already burned into a video.

The cover applied downstream is a full-width ffmpeg ``drawbox``, so only the
vertical extent matters -- no OCR, no per-character boxes. Detection samples
frames, measures how "text-like" every pixel row looks, and picks the band that
is both strong in absolute terms and clearly above the rest of the frame.
"""

import json
import os

import cv2
import numpy as np

from core.utils import rprint

# Bump whenever the algorithm or the constants below change, so cached results
# from an older detector are recomputed instead of silently reused.
DETECTOR_VERSION = 4

CACHE_FILE = "output/log/hardsub_region.json"
DEBUG_IMAGE = "output/log/hardsub_debug.png"

# Frames are analysed at this width so every threshold below is resolution
# independent and comparable across videos.
WORK_WIDTH = 640
# Sobel magnitude above which a pixel counts as a strong vertical edge. Absolute
# (not a per-frame percentile) so `strength` stays comparable between videos.
EDGE_THRESH = 120.0
# A row counts as "has text" when its edge fraction clears this floor and also
# clears BG_MULTIPLIER times the frame's own background level. A purely fixed
# threshold sits too close to the background of detailed footage, which makes
# almost every row a candidate; the background term keeps the two separated.
MIN_ACTIVE_THRESH = 0.10
BG_MULTIPLIER = 2.0
# Ignore the outer 5% of columns on each side (borders, letterbox artefacts).
SIDE_MARGIN_RATIO = 0.05
# Skip intro/outro, where credits produce text that is not subtitles.
EDGE_TRIM_RATIO = 0.05
MIN_VALID_FRAMES = 20

# Once a band core is found, its edges grow over neighbouring rows whose peak
# (95th percentile) score reaches this multiple of the active threshold.
# Subtitle blocks vary between one and several lines, so the topmost line may
# only appear in a small share of frames -- its *presence* is then buried in the
# background, but it still lights up strongly whenever it is on screen.
PEAK_MULTIPLIER = 2.0

# Row-score std at which a band is considered fully "changing over time". Must
# stay well above the std of ordinary footage, otherwise background noise earns
# nearly full credit and the weighting stops discriminating.
VAR_REF = 0.15
# A perfectly static band (logo, channel bug) keeps this share of its score: it
# can still win when nothing else is present, but always loses to real subtitles.
STATIC_FLOOR = 0.25
# Calibration references for the two confidence gates. Both are absolute, so a
# video with no hardcoded subtitles cannot score high just by being the best of
# a bad field.
ABS_STRENGTH_REF = 0.12
CONTRAST_REF = 3.0

# The band spans the full ink extent, tails included, because it has to *cover*
# the old subtitles. Glyph height is a different question -- it drives the font
# size of the replacement line -- so it is measured separately as the rows whose
# peak score reaches this share of the band's own peak.
TEXT_CORE_RATIO = 0.5


def _sample_gray_frames(cap, sample_frames):
    """Evenly sampled grayscale frames, downscaled to WORK_WIDTH."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        trim = int(total * EDGE_TRIM_RATIO)
        indices = np.linspace(trim, max(trim, total - trim - 1), sample_frames)
        indices = sorted(set(int(i) for i in indices))
    else:
        # Some containers report no frame count; fall back to sequential reads.
        indices = None

    frames = []
    if indices is not None:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    else:
        while len(frames) < sample_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)

    grays = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        work_h = max(1, round(h * WORK_WIDTH / w))
        grays.append(cv2.resize(gray, (WORK_WIDTH, work_h), interpolation=cv2.INTER_AREA))
    return grays


def _row_scores(gray):
    """Per-row fraction of columns carrying a strong vertical edge."""
    sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    margin = int(gray.shape[1] * SIDE_MARGIN_RATIO)
    core = sobel[:, margin:gray.shape[1] - margin]
    return (core > EDGE_THRESH).mean(axis=1)


def _smooth(arr, window=3):
    # Replicate the edges so the topmost/bottommost rows are not damped -- the
    # subtitle band very often touches the bottom of the frame.
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _group_bands(mask, bridge_gap):
    """Contiguous runs of True, merging runs separated by <= bridge_gap rows."""
    bands, start, gap = [], None, 0
    for i, active in enumerate(mask):
        if active:
            if start is None:
                start = i
            elif gap:
                gap = 0
            end = i
        elif start is not None:
            gap += 1
            if gap > bridge_gap:
                bands.append((start, end))
                start, gap = None, 0
    if start is not None:
        bands.append((start, end))
    return bands


def _grow_band(band, grow_bands):
    """Widen a seed band to the extent of any grow band it overlaps.

    Growth is anchored to a seed, so background rows that happen to clear the
    peak threshold on their own can never form a band by themselves.
    """
    top, bottom = band
    for g_top, g_bottom in grow_bands:
        if g_top <= bottom and g_bottom >= top:
            top, bottom = min(top, g_top), max(bottom, g_bottom)
    return top, bottom


def _text_core(peak, top, bottom):
    """Rows inside a band where the glyphs actually are.

    The band tapers off at both ends -- antialiasing, outlines and descenders
    all light up rows that carry no real glyph body. Cutting at a share of the
    band's own peak leaves the dense part, which is what a reader perceives as
    the text height.
    """
    band_peak = peak[top:bottom + 1]
    rows = np.where(band_peak >= TEXT_CORE_RATIO * band_peak.max())[0]
    if len(rows) == 0:
        return top, bottom
    return top + int(rows[0]), top + int(rows[-1])


def _analyze(grays, cfg):
    """Score every candidate band; return the best one in work-resolution rows."""
    scores = np.array([_row_scores(g) for g in grays])  # (n_frames, work_h)
    active_thresh = max(MIN_ACTIVE_THRESH, BG_MULTIPLIER * float(np.median(scores)))
    strength = _smooth(scores.mean(axis=0))
    presence = _smooth((scores >= active_thresh).mean(axis=0))
    variability = _smooth(scores.std(axis=0))

    var_norm = np.clip(variability / VAR_REF, 0.0, 1.0)
    row_score = presence * strength * (STATIC_FLOOR + (1 - STATIC_FLOOR) * var_norm)

    work_h = scores.shape[1]
    min_presence = float(cfg.get("min_presence", 0.15))
    min_band = float(cfg.get("min_band_ratio", 0.02)) * work_h
    max_band = float(cfg.get("max_band_ratio", 0.35)) * work_h

    bridge_gap = max(1, int(0.02 * work_h))
    bands = _group_bands(presence >= min_presence, bridge_gap)
    peak = _smooth(np.percentile(scores, 95, axis=0))
    grow_bands = _group_bands(peak >= active_thresh * PEAK_MULTIPLIER, bridge_gap)
    bands = [_grow_band(b, grow_bands) for b in bands]
    candidates = [b for b in bands if min_band <= (b[1] - b[0] + 1) <= max_band]
    if not candidates:
        return None, {"reason": "no band passed the size filter", "bands": len(bands),
                      "active_thresh": round(active_thresh, 4), "work_height": work_h}

    top, bottom = max(candidates, key=lambda b: row_score[b[0]:b[1] + 1].mean())
    band_row_score = float(row_score[top:bottom + 1].mean())
    band_strength = float(strength[top:bottom + 1].mean())
    median_row_score = float(np.median(row_score))
    core_top, core_bottom = _text_core(peak, top, bottom)

    abs_conf = float(np.clip(band_strength / ABS_STRENGTH_REF, 0.0, 1.0))
    ratio = band_row_score / max(median_row_score, 1e-6)
    contrast_conf = float(np.clip((ratio - 1.0) / CONTRAST_REF, 0.0, 1.0))
    confidence = min(abs_conf, contrast_conf)

    stats = {
        "band_strength": round(band_strength, 4),
        "band_row_score": round(band_row_score, 5),
        "median_row_score": round(median_row_score, 5),
        "contrast_ratio": round(ratio, 2),
        "abs_conf": round(abs_conf, 3),
        "contrast_conf": round(contrast_conf, 3),
        "active_thresh": round(active_thresh, 4),
        "work_height": work_h,
        "core_rows": [core_top, core_bottom],
    }
    return (top, bottom, confidence, core_top, core_bottom), stats


def _cache_key(video_path, cfg):
    # The cached band is already padded and confidence-filtered, so the settings
    # that produced it belong in the key -- otherwise retuning them in config
    # would silently keep returning the old band.
    st = os.stat(video_path)
    settings = json.dumps(cfg, sort_keys=True, default=str)
    return (f"{os.path.basename(video_path)}|{st.st_size}|{int(st.st_mtime)}"
            f"|{settings}|v{DETECTOR_VERSION}")


def _read_cache(key):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get(key)
    except (OSError, ValueError):
        return None


def _write_cache(key, value):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        data = {}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except ValueError:
                data = {}
        # Entries from superseded detector versions can never be read again.
        suffix = f"|v{DETECTOR_VERSION}"
        data = {k: v for k, v in data.items() if k.endswith(suffix)}
        data[key] = value
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass  # caching is an optimisation, never a failure


def detect_subtitle_band(video_path, height, cfg=None, use_cache=True):
    """Row band of hardcoded subtitles, in full-resolution coordinates.

    Returns ``{"y", "height", "text_height", "confidence"}`` or None when nothing
    is found confidently enough, in which case the caller should fall back to
    fixed ratios. ``height`` spans everything that must be covered; the smaller
    ``text_height`` is the glyph height the replacement subtitles should match.
    """
    cfg = cfg or {}
    key = _cache_key(video_path, cfg)
    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached.get("band")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        rprint("[bold yellow]⚠️ Could not open video for subtitle detection.[/bold yellow]")
        return None
    try:
        grays = _sample_gray_frames(cap, int(cfg.get("sample_frames", 100)))
    finally:
        cap.release()

    if len(grays) < MIN_VALID_FRAMES:
        rprint(f"[bold yellow]⚠️ Only {len(grays)} frames sampled, too few to detect subtitles.[/bold yellow]")
        return None

    result, stats = _analyze(grays, cfg)
    min_confidence = float(cfg.get("min_confidence", 0.35))

    band = None
    if result is not None:
        top, bottom, confidence, core_top, core_bottom = result
        if confidence >= min_confidence:
            work_h = stats["work_height"]
            pad = int(float(cfg.get("padding_ratio", 0.006)) * work_h)
            top = max(0, top - pad)
            bottom = min(work_h - 1, bottom + pad)
            scale = height / work_h
            y = int(round(top * scale))
            band = {
                "y": max(0, y),
                "height": max(1, min(height - y, int(round((bottom - top + 1) * scale)))),
                "text_height": max(1, int(round((core_bottom - core_top + 1) * scale))),
                "confidence": round(confidence, 3),
            }
        else:
            rprint(f"[bold yellow]⚠️ Hardcoded subtitle detection confidence {confidence:.2f} "
                   f"< {min_confidence}. Stats: {stats}[/bold yellow]")

    _write_cache(key, {"band": band, "stats": stats})
    return band


def _preview_frame(sample, band, height):
    """Paint the bar and the matched font height onto a frame, as burning would.

    Burning a full feature to inspect a bar costs an encode of the whole video,
    so the preview has to answer "is the bar too big" on its own.
    """
    from core._7_sub_into_vid import PLAY_RES_Y, _matched_font_size
    from core.utils.config_utils import load_key

    cfg = load_key("cover_hardcoded_subtitles") or {}
    y, bar_h = band["y"], band["height"]
    # Darkened rather than filled: burning paints it solid black, but the point
    # of the preview is to see whether the old text falls inside the bar.
    sample[y:y + bar_h, :] = (sample[y:y + bar_h, :] * 0.35).astype(sample.dtype)

    font_size = _matched_font_size(band["text_height"], height, cfg)
    em_px = round(font_size / PLAY_RES_Y * height)
    center = y + bar_h // 2
    top, bottom = center - em_px // 2, center + em_px // 2
    cv2.rectangle(sample, (0, top), (sample.shape[1] - 1, bottom), (0, 255, 255), 2)
    cv2.rectangle(sample, (0, y), (sample.shape[1] - 1, y + bar_h - 1), (0, 0, 255), 2)
    print(f"bar            : {bar_h}px ({bar_h / height:.1%} of the frame)")
    print(f"matched font   : FontSize={font_size} -> {em_px}px em "
          f"vs {band['text_height']}px of source glyphs")
    return sample


def _debug_main(video_path):
    """Print the detected band and dump a sample frame with it outlined."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    grays = _sample_gray_frames(cap, 100)
    cap.release()

    if len(grays) < MIN_VALID_FRAMES:
        raise SystemExit(f"Only {len(grays)} frames sampled")

    result, stats = _analyze(grays, {})
    print(f"frames sampled : {len(grays)}")
    print(f"stats          : {json.dumps(stats, indent=2)}")

    band = detect_subtitle_band(video_path, height, use_cache=False)
    print(f"band           : {band}")
    if not band:
        return

    # Pick the sampled frame with the busiest detected band, so the dumped frame
    # is one that actually shows a subtitle rather than a gap between lines.
    work_h = stats["work_height"]
    top = int(band["y"] * work_h / height)
    bottom = int((band["y"] + band["height"]) * work_h / height)
    busiest = max(range(len(grays)), key=lambda i: _row_scores(grays[i])[top:bottom].mean())

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    trim = int(total * EDGE_TRIM_RATIO)
    indices = sorted(set(int(i) for i in np.linspace(trim, max(trim, total - trim - 1), 100)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[busiest])
    ok, sample = cap.read()
    cap.release()

    if ok:
        os.makedirs(os.path.dirname(DEBUG_IMAGE), exist_ok=True)
        cv2.imwrite(DEBUG_IMAGE, _preview_frame(sample, band, height))
        print(f"debug image    : {DEBUG_IMAGE} (frame {indices[busiest]})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m core.utils.hardsub_detect <video>")
    _debug_main(sys.argv[1])
