"""Locate the row band and column extent occupied by subtitles already burned
into a video.

The cover applied downstream is an ffmpeg ``drawbox`` sized to the detected
band -- no OCR, no per-character boxes. Detection samples frames, measures how
"text-like" every pixel row/column looks, and picks the band that is both
strong in absolute terms and clearly above the rest of the frame.
"""

import json
import os
import zlib

import cv2
import numpy as np

from core.utils import rprint

# Bump whenever the algorithm or the constants below change, so cached results
# from an older detector are recomputed instead of silently reused.
DETECTOR_VERSION = 7

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

# --- Horizontal extent -------------------------------------------------------
# Share of a band's rows a column must light up, in one frame, to count as
# carrying text there. Kept separate from (and below) the row threshold: a bar
# that ends up too narrow leaves the old subtitles peeking out the sides, which
# is worse than one that is slightly too wide.
COL_ACTIVE_THRESH = 0.10
# Share of the frame width the column profile is smoothed over, to close the
# gaps between strokes and words, and the gap bridged when grouping runs.
COL_SMOOTH_RATIO = 0.015
COL_BRIDGE_RATIO = 0.04
# How often a column has to carry text before it counts as part of the line:
# an absolute floor, and a multiple of the background level of the same profile.
# Lines vary in length, so the floor stays low enough to keep the ends of the
# longer ones -- `width_padding_ratio` in config then adds the final margin.
COL_PRESENCE_FLOOR = 0.15
COL_BG_MULTIPLIER = 2.5
# Runs narrower than this share of the frame are background speckle, not a line.
COL_MIN_RUN_RATIO = 0.02
# Below this many frames carrying text the extent is not measured at all and the
# bar falls back to the full frame width.
MIN_TEXT_FRAMES = 5

# --- Horizontal extent over time ---------------------------------------------
# One extent for the whole video has to fit the longest line there is, so every
# short line gets a bar far wider than its text. Sampling on a fixed grid gives
# the extent of whatever is on screen at each moment instead. The grid is not
# the transcript's: those cues are timed from the audio and routinely run tens
# of seconds, while the burned-in text changes several times inside one of them.
# Repetition cannot be used to separate text from background at a single moment,
# so each frame is thresholded against its own background instead.
SPAN_STEP = 1.0
# A column must reach this share of the band's rows to carry text. Absolute, so
# a frame showing no text at all yields no span rather than a background one.
SPAN_FLOOR = 0.12
SPAN_BG_MULTIPLIER = 2.5
# Percentile of the column profile taken as its background level. Low enough
# that a line covering most of the frame does not raise its own threshold.
SPAN_BG_PERCENTILE = 25
# Levels for growing the span outwards from the strongest run, over the fainter
# ends of the same line, and the gap that growth is allowed to bridge.
SPAN_WEAK_FLOOR = 0.05
SPAN_WEAK_MULTIPLIER = 1.5
SPAN_JOIN_RATIO = 0.06
# A span narrower than this share of the frame is background speckle rather than
# a line of subtitles, and putting a bar on it would only add a flicker.
SPAN_MIN_WIDTH_RATIO = 0.06
SPAN_CACHE_FILE = "output/log/hardsub_spans.json"


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


def _col_scores(gray, top, bottom):
    """Per-column fraction of rows, within a row band, carrying a strong vertical edge."""
    sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    band = sobel[top:bottom + 1, :]
    return (band > EDGE_THRESH).mean(axis=0)


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


def _text_columns(grays, top, bottom, text_frames):
    """Left/right bounds of the columns the subtitles occupy within a row band.

    Neither "every column ever active" nor a trimmed per-frame extent survives
    detailed footage: background inside the band lights up columns right out to
    the frame edges in *every* frame, so there is no outlier to trim. What
    separates the two is repetition -- the subtitles sit in the same columns
    every time they appear, while background detail moves. So each column is
    scored by how often it carries text across the frames that have any, and the
    level of the frame's own background sets the bar it has to clear.
    """
    work_w = grays[0].shape[1]
    hits = np.zeros(work_w)
    frames_used = 0
    for gray, has_text in zip(grays, text_frames):
        if not has_text:
            continue
        hits += _col_scores(gray, top, bottom) >= COL_ACTIVE_THRESH
        frames_used += 1
    if frames_used < MIN_TEXT_FRAMES:
        return None

    # Glyphs only produce edges at their strokes, so a raw column profile is a
    # comb with gaps between strokes and words. Smoothing (over an odd window, to
    # keep the profile aligned with the column index) turns a line of text into
    # the one plateau the extent is then measured from.
    presence = _smooth(hits / frames_used,
                       window=max(3, int(COL_SMOOTH_RATIO * work_w) | 1))
    margin = int(work_w * SIDE_MARGIN_RATIO)
    presence[:margin] = 0.0
    presence[work_w - margin:] = 0.0

    # A centred line leaves most columns as background, so the median is a fair
    # estimate of it -- the same trick the row threshold uses.
    thresh = max(COL_PRESENCE_FLOOR, COL_BG_MULTIPLIER * float(np.median(presence)))
    groups = _group_bands(presence >= thresh, max(2, int(COL_BRIDGE_RATIO * work_w)))
    groups = [g for g in groups if (g[1] - g[0] + 1) >= COL_MIN_RUN_RATIO * work_w]
    if not groups:
        return None
    # The text is the strongest run; stray background runs that cleared the
    # threshold are not allowed to drag the edges outwards.
    left, right = max(groups, key=lambda g: presence[g[0]:g[1] + 1].sum())
    if right <= left:
        return None
    return int(left), int(right)


def _frame_span(gray, top, bottom):
    """Left/right columns carrying text in a single frame, or None when it is blank.

    Same profile the video-wide measurement uses, but scored within one frame:
    the threshold is the frame's own background level inside the band, and the
    absolute floor is what makes a frame with no subtitle return nothing at all
    instead of the extent of whatever detail happens to sit in those rows.

    Background is read off a low percentile rather than the median, because a
    line of subtitles routinely covers more than half the frame -- the median of
    such a profile is the text itself, and thresholding against it keeps only
    the few strongest words. What remains of the line is then recovered by
    growing the strongest run outwards at a much lower level: the ends of a line
    sitting over a bright background produce far weaker edges than its middle,
    but they still have to end up under the bar.
    """
    work_w = gray.shape[1]
    cols = _smooth(_col_scores(gray, top, bottom),
                   window=max(3, int(COL_SMOOTH_RATIO * work_w) | 1))
    margin = int(work_w * SIDE_MARGIN_RATIO)
    cols[:margin] = 0.0
    cols[work_w - margin:] = 0.0
    if cols.max() < SPAN_FLOOR:
        return None

    background = float(np.percentile(cols, SPAN_BG_PERCENTILE))
    thresh = max(SPAN_FLOOR, SPAN_BG_MULTIPLIER * background)
    groups = _group_bands(cols >= thresh, max(2, int(COL_BRIDGE_RATIO * work_w)))
    groups = [g for g in groups if (g[1] - g[0] + 1) >= COL_MIN_RUN_RATIO * work_w]
    if not groups:
        return None

    left, right = max(groups, key=lambda g: cols[g[0]:g[1] + 1].sum())
    weak = max(SPAN_WEAK_FLOOR, SPAN_WEAK_MULTIPLIER * background)
    join = int(SPAN_JOIN_RATIO * work_w)
    for step in (-1, 1):
        edge, gap = (left if step < 0 else right), 0
        i = edge + step
        while 0 <= i < work_w:
            if cols[i] >= weak:
                edge, gap = i, 0
            else:
                gap += 1
                if gap > join:
                    break
            i += step
        if step < 0:
            left = edge
        else:
            right = edge
    if right - left + 1 < SPAN_MIN_WIDTH_RATIO * work_w:
        return None
    return int(left), int(right)


def _span_cache_key(video_path, y, band_height, step):
    st = os.stat(video_path)
    return (f"{os.path.basename(video_path)}|{st.st_size}|{int(st.st_mtime)}"
            f"|{y},{band_height},{step}|v{DETECTOR_VERSION}")


def measure_span_timeline(video_path, y, band_height, step=SPAN_STEP, use_cache=True):
    """Horizontal extent of the burned-in text through the video, in full-res px.

    Returns ``[(t, (left, right) or None), ...]`` on a ``step``-second grid: the
    columns the text occupies at that moment, or None where the band holds no
    text at all. Seeking costs ~30ms a sample, so a grid this dense is a few
    seconds of work on a video that then takes minutes to encode, and it is
    cached against the file and the band it was measured for.
    """
    key = _span_cache_key(video_path, y, band_height, step)
    if use_cache:
        cached = _read_cache(key, SPAN_CACHE_FILE)
        if cached is not None:
            return [(t, tuple(span) if span else None) for t, span in cached]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        rprint("[bold yellow]⚠️ Could not open video to measure the subtitle width.[/bold yellow]")
        return []

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frames / fps if fps > 0 and frames > 0 else 0
        if not width or not height or duration <= 0:
            return []

        work_h = max(1, round(height * WORK_WIDTH / width))
        top = max(0, int(round(y * work_h / height)))
        bottom = min(work_h - 1, int(round((y + band_height - 1) * work_h / height)))
        scale_x = width / WORK_WIDTH

        timeline = []
        for t in np.arange(step / 2, duration, step):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (WORK_WIDTH, work_h), interpolation=cv2.INTER_AREA)
            span = _frame_span(gray, top, bottom)
            timeline.append((round(float(t), 3), (int(round(span[0] * scale_x)),
                                                  int(round((span[1] + 1) * scale_x)))
                             if span else None))
    finally:
        cap.release()

    _write_cache(key, timeline, SPAN_CACHE_FILE)
    return timeline


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
    # Same "is a subtitle on screen" test the row logic uses for presence, but
    # per frame rather than averaged, so only frames showing text are measured.
    text_frames = scores[:, core_top:core_bottom + 1].mean(axis=1) >= active_thresh
    col_bounds = _text_columns(grays, top, bottom, text_frames)

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
        "col_bounds": col_bounds,
        "text_frames": int(text_frames.sum()),
    }
    return (top, bottom, confidence, core_top, core_bottom, col_bounds), stats


def _cache_key(video_path, cfg):
    # The cached band is already padded and confidence-filtered, so the settings
    # that produced it belong in the key -- otherwise retuning them in config
    # would silently keep returning the old band.
    st = os.stat(video_path)
    settings = json.dumps(cfg, sort_keys=True, default=str)
    return (f"{os.path.basename(video_path)}|{st.st_size}|{int(st.st_mtime)}"
            f"|{settings}|v{DETECTOR_VERSION}")


def _read_cache(key, path=CACHE_FILE):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get(key)
    except (OSError, ValueError):
        return None


def _write_cache(key, value, path=CACHE_FILE):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except ValueError:
                data = {}
        # Entries from superseded detector versions can never be read again.
        suffix = f"|v{DETECTOR_VERSION}"
        data = {k: v for k, v in data.items() if k.endswith(suffix)}
        data[key] = value
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass  # caching is an optimisation, never a failure


def detect_subtitle_band(video_path, width, height, cfg=None, use_cache=True):
    """Row band of hardcoded subtitles, in full-resolution coordinates.

    Returns ``{"y", "height", "text_height", "x", "width", "confidence"}`` or
    None when nothing is found confidently enough, in which case the caller
    should fall back to fixed ratios. ``y``/``height`` span everything that must
    be covered vertically; ``x``/``width`` are the horizontal extent of the text
    rather than the full frame width, measured from where text repeatedly
    appears and widened by ``width_padding_ratio``. The smaller ``text_height``
    is the glyph height the replacement subtitles should match.
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
        top, bottom, confidence, core_top, core_bottom, col_bounds = result
        if confidence >= min_confidence:
            work_h = stats["work_height"]
            pad = int(float(cfg.get("padding_ratio", 0.006)) * work_h)
            top = max(0, top - pad)
            bottom = min(work_h - 1, bottom + pad)
            scale_y = height / work_h
            y = int(round(top * scale_y))

            work_w = WORK_WIDTH
            left, right = col_bounds if col_bounds else (0, work_w - 1)
            col_pad = int(float(cfg.get("width_padding_ratio", 0.03)) * work_w)
            left = max(0, left - col_pad)
            right = min(work_w - 1, right + col_pad)
            scale_x = width / work_w
            x = int(round(left * scale_x))

            band = {
                "y": max(0, y),
                "height": max(1, min(height - y, int(round((bottom - top + 1) * scale_y)))),
                "text_height": max(1, int(round((core_bottom - core_top + 1) * scale_y))),
                "x": max(0, x),
                "width": max(1, min(width - x, int(round((right - left + 1) * scale_x)))),
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
    x, bar_w = band.get("x", 0), band.get("width", sample.shape[1])
    # Darkened rather than filled: burning paints it solid black, but the point
    # of the preview is to see whether the old text falls inside the bar.
    sample[y:y + bar_h, x:x + bar_w] = (sample[y:y + bar_h, x:x + bar_w] * 0.35).astype(sample.dtype)

    font_size = _matched_font_size(band["text_height"], height, cfg)
    em_px = round(font_size / PLAY_RES_Y * height)
    center = y + bar_h // 2
    top, bottom = center - em_px // 2, center + em_px // 2
    cv2.rectangle(sample, (x, top), (x + bar_w - 1, bottom), (0, 255, 255), 2)
    cv2.rectangle(sample, (x, y), (x + bar_w - 1, y + bar_h - 1), (0, 0, 255), 2)
    print(f"bar            : {bar_w}x{bar_h}px ({bar_h / height:.1%} of the frame height)")
    print(f"matched font   : FontSize={font_size} -> {em_px}px em "
          f"vs {band['text_height']}px of source glyphs")
    return sample


def _debug_main(video_path):
    """Print the detected band and dump a sample frame with it outlined."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    grays = _sample_gray_frames(cap, 100)
    cap.release()

    if len(grays) < MIN_VALID_FRAMES:
        raise SystemExit(f"Only {len(grays)} frames sampled")

    result, stats = _analyze(grays, {})
    print(f"frames sampled : {len(grays)}")
    print(f"stats          : {json.dumps(stats, indent=2)}")

    band = detect_subtitle_band(video_path, width, height, use_cache=False)
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
