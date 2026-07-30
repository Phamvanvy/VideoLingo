import math, os, re, subprocess, time
from core._1_ytdlp import find_video_files
import cv2
import numpy as np
import platform
from core.utils import *

SRC_FONT_SIZE = 15
TRANS_FONT_SIZE = 17
FONT_NAME = 'Arial'
TRANS_FONT_NAME = 'Arial'

# Linux need to install google noto fonts: apt-get install fonts-noto
if platform.system() == 'Linux':
    FONT_NAME = 'NotoSansCJK-Regular'
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
# Mac OS has different font names
elif platform.system() == 'Darwin':
    FONT_NAME = 'Arial Unicode MS'
    TRANS_FONT_NAME = 'Arial Unicode MS'

SRC_FONT_COLOR = '&HFFFFFF'
SRC_OUTLINE_COLOR = '&H000000'
SRC_OUTLINE_WIDTH = 1
SRC_SHADOW_COLOR = '&H80000000'
TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1
TRANS_BACK_COLOR = '&H33000000'
TRANS_MARGIN_V = 27

# libass script height ffmpeg uses when converting an srt to ass, all margins are in this scale
PLAY_RES_Y = 288

OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/output_sub.mp4"
SRC_SRT = f"{OUTPUT_DIR}/src.srt"
TRANS_SRT = f"{OUTPUT_DIR}/trans.srt"
    
def check_gpu_available():
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        return 'h264_nvenc' in result.stdout
    except:
        return False

def load_key_safe(key, default=None):
    """Config files written before a key existed should not break the merge."""
    try:
        return load_key(key)
    except KeyError:
        return default


# Audio codecs an MP4 container can carry as-is. Anything else (opus, vorbis,
# wmav2 -- reachable through the webm/mkv/wmv inputs VideoLingo accepts) has to
# be converted even though the filter chain never touches the audio.
_MP4_SAFE_AUDIO = {"aac", "mp3", "alac", "ac3", "eac3"}


def _ffprobe(video_file, *args):
    """Single ffprobe field, or None when it is missing or ffprobe is absent."""
    cmd = ['ffprobe', '-v', 'error', *args, '-of', 'default=nw=1:nk=1', video_file]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines[0] if lines and lines[0] != "N/A" else None


def source_video_bitrate(video_file):
    """Bitrate of the source video stream in bits per second, or None.

    Some containers report no per-stream bitrate, so the container total minus a
    rough audio allowance stands in for it.
    """
    stream = _ffprobe(video_file, '-select_streams', 'v:0', '-show_entries', 'stream=bit_rate')
    if stream and stream.isdigit() and int(stream) > 0:
        return int(stream)
    total = _ffprobe(video_file, '-show_entries', 'format=bit_rate')
    if total and total.isdigit() and int(total) > 200_000:
        return int(total) - 192_000
    return None


def video_encoder_args(video_file):
    """-c:v flags that re-encode at roughly the bitrate of the source.

    Burning subtitles forces a full re-encode, and both encoders otherwise pick
    a quality that has nothing to do with the input -- h264_nvenc in particular
    falls back to its own 2 Mbps default, which visibly softens anything shot
    higher. Targeting the measured source bitrate keeps the output looking like
    the input; the headroom on maxrate absorbs the sharp glyph edges the burn
    adds, which cost more bits than the picture underneath them.
    """
    gpu = load_key("ffmpeg_gpu")
    if gpu:
        rprint("[bold green]will use GPU acceleration.[/bold green]")
    base = ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr'] if gpu else \
           ['-c:v', 'libx264', '-preset', 'medium']

    bitrate = source_video_bitrate(video_file)
    if bitrate:
        rprint(f"[bold green]📊 Matching the source video bitrate: {bitrate / 1e6:.2f} Mbps.[/bold green]")
        return base + ['-b:v', str(bitrate),
                       '-maxrate', str(int(bitrate * 1.5)),
                       '-bufsize', str(bitrate * 3)]

    rprint("[bold yellow]⚠️ Could not read the source bitrate; falling back to constant quality.[/bold yellow]")
    # -b:v 0 is what actually arms nvenc's constant-quality mode; without it
    # -cq is ignored and the 2 Mbps default comes back.
    return base + (['-cq', '21', '-b:v', '0'] if gpu else ['-crf', '20'])


def audio_encoder_args(video_file):
    """-c:a flags that leave the source audio alone whenever MP4 allows it."""
    codec = _ffprobe(video_file, '-select_streams', 'a:0', '-show_entries', 'stream=codec_name')
    if codec in _MP4_SAFE_AUDIO:
        return ['-c:a', 'copy']
    return ['-c:a', 'aac', '-b:a', '192k']

_SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def _srt_time_ranges(srt_path):
    """(start, end) seconds for every cue in an SRT file, in file order."""
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return []

    def to_seconds(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    ranges = []
    for h1, m1, s1, ms1, h2, m2, s2, ms2 in _SRT_TIME_RE.findall(content):
        start, end = to_seconds(h1, m1, s1, ms1), to_seconds(h2, m2, s2, ms2)
        if end > start:
            ranges.append((start, end))
    return ranges


def _srt_cues(srt_path):
    """(start, end, text) for every cue in an SRT file, in file order."""
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read().replace("\r\n", "\n")
    except OSError:
        return []

    cues = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line for line in block.split("\n") if line.strip()]
        for i, line in enumerate(lines):
            match = _SRT_TIME_RE.search(line)
            if not match:
                continue
            h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
            start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000
            end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000
            if end > start:
                cues.append((start, end, " ".join(lines[i + 1:]).strip()))
            break
    return cues


def _merge_ranges(ranges, gap):
    """Sorted ranges with anything closer together than `gap` fused into one."""
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


# One `between(t,...)` term costs ~30 characters, and the whole filter chain has
# to fit in a single command-line argument (Windows caps the command line at
# 32767 characters). Staying well under that leaves room for the rest of the
# chain no matter how long the video is.
MAX_ENABLE_SEGMENTS = 300
# Gaps tried in order until the segment count fits the budget. A feature-length
# video has thousands of cues, so instead of giving up on gating it the bar is
# simply held on across the shorter pauses -- it still switches off through
# intros, music and any long stretch with no dialogue at all.
MERGE_GAPS = (0.35, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0, 90.0)


def _cover_enable_expr(srt_paths, cfg, pad=0.15):
    """ffmpeg `enable` expression covering only the time subtitles are on screen,
    so the bar does not sit over the picture between lines.

    Every SRT that matters is unioned together: the source timing decides when
    the burned-in subtitles need covering, and the timing of the replacement
    line decides when it needs the bar behind it -- dubbing shifts the latter
    away from the former. Ranges are padded slightly because the hardcoded
    subtitles' fade in/out rarely lines up to the millisecond with the
    transcript timing them.
    """
    if not cfg.get("time_gate", True):
        return None

    ranges = []
    for path in srt_paths:
        ranges.extend(_srt_time_ranges(path))
    if not ranges:
        rprint("[bold yellow]⚠️ No subtitle timings found; the cover bar stays on "
               "for the whole video.[/bold yellow]")
        return None

    padded = sorted((max(0.0, s - pad), e + pad) for s, e in ranges)
    for gap in MERGE_GAPS:
        merged = _merge_ranges(padded, gap)
        if len(merged) <= MAX_ENABLE_SEGMENTS:
            if gap != MERGE_GAPS[0]:
                rprint(f"[bold yellow]⚠️ {len(padded)} subtitle cues is too many to gate "
                       f"the cover bar one by one; holding it on across gaps shorter "
                       f"than {gap:g}s ({len(merged)} segments).[/bold yellow]")
            return "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in merged)

    rprint("[bold yellow]⚠️ Subtitles are too dense to gate the cover bar; "
           "leaving it on for the whole video.[/bold yellow]")
    return None


def _fixed_bar(cfg, height):
    """Bar geometry from the static ratios in config."""
    height_ratio = float(cfg.get("height_ratio", 0.10))
    offset_ratio = float(cfg.get("bottom_offset_ratio", 0.0))
    bar_height = max(1, int(height * height_ratio))
    y = max(0, height - bar_height - int(height * offset_ratio))
    return y, bar_height


def _matched_font_size(text_height, height, cfg):
    """libass FontSize whose glyphs are as tall as the text being covered.

    CJK glyphs fill nearly their whole em box, so the measured ink height of the
    hardcoded subtitles doubles as the em size that matches them. ``font_scale``
    then trims it slightly so the replacement never draws taller than the line
    it replaces.
    """
    size = round(text_height / height * PLAY_RES_Y * float(cfg.get("font_scale", 0.95)))
    lo, hi = int(cfg.get("min_font_size", 10)), int(cfg.get("max_font_size", 30))
    return int(min(hi, max(lo, size)))


# PlayResX of the style ffmpeg builds when it converts an srt to ass, and the
# side margins of that style -- together they set the width libass wraps at.
PLAY_RES_X = 384
ASS_SIDE_MARGIN = 10

# Every bar costs ~90 characters of the one command-line argument the whole
# filter chain has to fit in (Windows caps that at 32767), and the subtitle
# filters come after it. Overrunning this budget falls back to the static bar.
MAX_BAR_CHARS = 16000
# Measured widths wobble by a few pixels between one sample and the next, which
# would cut the timeline into a separate bar per sample. Rounding up to a grid
# makes a steady line one bar; the coarser steps are there for a video whose
# lines change often enough that the fine grid overruns the budget above.
WIDTH_QUANTA = (32, 64, 128, 256)
# Pieces shorter than this are absorbed into the bar before them: at a fifth of
# a second a width change is a flicker, not something the eye reads as a bar.
MIN_BAR_SEGMENT = 0.4

_FONT_FILES = {"Windows": "arial.ttf", "Darwin": "Arial Unicode.ttf",
               "Linux": "NotoSansCJK-Regular.ttc"}
_font_cache = {}


def _text_measurer(em_px):
    """Callable giving the pixel width a string renders to at this em size.

    Taken from the real font file when it can be loaded, so the bar matches what
    libass will actually draw; the fallback assumes the average Latin glyph is
    half an em wide and a CJK one a full em.
    """
    px = max(1, int(round(em_px)))
    if px not in _font_cache:
        try:
            from PIL import ImageFont
            _font_cache[px] = ImageFont.truetype(
                _FONT_FILES.get(platform.system(), "arial.ttf"), px)
        except Exception:
            _font_cache[px] = None
    font = _font_cache[px]
    if font is not None:
        return lambda text: float(font.getlength(text))
    return lambda text: sum(1.0 if ord(c) > 0x2E80 else 0.5 for c in text) * px


def _rendered_text_width(text, font_size, height, max_width):
    """Width of the widest line libass draws for `text`, wrapping included."""
    measure = _text_measurer(font_size / PLAY_RES_Y * height)
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}" if current else word
        if current and measure(trial) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    return min(max_width, max(measure(line) for line in lines))


def _drawbox(x, y, w, h, color, opacity, gate=None):
    box = f"drawbox=x={x}:y={y}:w={w}:h={h}:color={color}@{opacity}:t=fill"
    return box + (f":enable='{gate}'" if gate else "")


def _width_requirements(video_file, y, bar_height, width, height, cfg,
                        text_srt, font_size, text_in_bar, pad=0.15):
    """(start, end, width) stretches the cover bar has to satisfy.

    Two things have to fit inside it. The burned-in line it hides is measured
    from the video on a fixed grid, and each sample claims a full step either
    side of itself, so a line whose text starts or ends between two samples is
    still covered and a single missed sample cannot punch a hole in the middle
    of a line. The replacement line is added on top of that whenever it is drawn
    inside the bar rather than below it -- there the bar is also its backdrop,
    and a backdrop narrower than the text it sits behind looks like a mistake.
    """
    from core.utils.hardsub_detect import SPAN_STEP, measure_span_timeline

    items = []
    timeline = measure_span_timeline(video_file, y, bar_height)
    center = width / 2
    side_pad = float(cfg.get("width_padding_ratio", 0.03)) * width
    for t, span in timeline:
        if span:
            # The bar stays centred on the frame, like the text drawn on it, so
            # it has to reach whichever side of the measured line is further out.
            needed = 2 * (max(center - span[0], span[1] - center) + side_pad)
            items.append((t - SPAN_STEP, t + SPAN_STEP, needed))

    if not items:
        return None

    if text_srt and text_in_bar:
        wrap_width = width - 2 * ASS_SIDE_MARGIN / PLAY_RES_X * width
        pad_px = 2 * float(cfg.get("line_width_padding_em", 0.6)) * font_size / PLAY_RES_Y * height
        for start, end, text in _srt_cues(text_srt):
            drawn = _rendered_text_width(text, font_size, height, wrap_width)
            items.append((start - pad, end + pad, drawn + pad_px))
    return items


def _bar_intervals(items, quantum, max_width):
    """Requirements resolved into disjoint (start, end, width) intervals.

    Neighbouring stretches must not be fused -- that would hand the shorter line
    the width of the longer one, which is the whole thing being avoided here. So
    the timeline is cut at every requirement boundary instead, and each piece
    takes the widest requirement covering it, rounded up to the quantum: where
    two lines overlap the bar spans both rather than cropping either.
    """
    bounds = sorted({t for start, end, _ in items for t in (start, end)})
    intervals = []
    for left, right in zip(bounds, bounds[1:]):
        active = [needed for start, end, needed in items if start < right and end > left]
        if not active:
            continue
        width = min(max_width, int(math.ceil(max(active) / quantum) * quantum))
        if intervals and intervals[-1][1] >= left and intervals[-1][2] == width:
            intervals[-1][1] = right  # same width as the piece before it
        else:
            intervals.append([left, right, width])
    return intervals


def _absorb_short(intervals, min_length=MIN_BAR_SEGMENT):
    """Slivers folded into the bar before them, at the wider of the two widths."""
    kept = []
    for start, end, width in intervals:
        # Only into the bar it touches: absorbing across a gap would hold the bar
        # up over a stretch that has nothing to cover.
        touches = bool(kept) and start - kept[-1][1] < 1e-6
        if touches and end - start < min_length:
            kept[-1][1] = end
            kept[-1][2] = max(kept[-1][2], width)
        elif touches and kept[-1][2] == width:
            kept[-1][1] = end
        else:
            kept.append([start, end, width])
    return kept


def _measured_bars(video_file, y, bar_height, width, height, cfg, text_srt, font_size,
                   text_in_bar, color, opacity):
    """One bar per stretch of subtitle, each only as wide as the text under it.

    Returns the comma-joined drawbox chain, or None when nothing could be
    measured or the result will not fit the command line, so the caller falls
    back to the single bar sized to the longest line in the whole video.
    """
    items = _width_requirements(video_file, y, bar_height, width, height, cfg,
                                text_srt, font_size, text_in_bar)
    if not items:
        rprint("[bold yellow]⚠️ Could not measure the subtitles line by line; using one "
               "bar for the whole video.[/bold yellow]")
        return None

    for quantum in WIDTH_QUANTA:
        intervals = _absorb_short(_bar_intervals(items, quantum, width))
        boxes = [_drawbox((width - box_w) // 2, y, box_w, bar_height, color, opacity,
                          f"between(t,{max(0.0, start):.3f},{end:.3f})")
                 for start, end, box_w in intervals]
        chain = ",".join(boxes)
        if len(chain) <= MAX_BAR_CHARS:
            widths = [box_w for _, _, box_w in intervals]
            on_screen = sum(end - start for start, end, _ in intervals)
            rprint(f"[bold green]🩹 Covering hardcoded subtitles with {len(boxes)} bars "
                   f"sized to the line under them, {min(widths)}-{max(widths)}px wide "
                   f"({bar_height}px tall, {color}@{opacity}, up {on_screen / 60:.1f} min "
                   f"in total).[/bold green]")
            return chain

    rprint("[bold yellow]⚠️ The subtitle width changes too often to give every line its "
           "own bar; using one bar for the whole video.[/bold yellow]")
    return None


def trans_backdrop_style(has_cover_bar):
    """force_style fragment deciding what sits behind the translated text.

    The opaque box (BorderStyle=4) is what makes the text readable when it is
    drawn straight onto the picture. Over a cover bar it only adds a second,
    darker rectangle on top of a black one, so there the outline is enough.
    """
    mode = (load_key_safe("cover_hardcoded_subtitles") or {}).get("text_backdrop", "auto")
    if mode == "box" or (mode == "auto" and not has_cover_bar):
        return f"BackColour={TRANS_BACK_COLOR},BorderStyle=4"
    return "BorderStyle=1"


def build_cover_bar(width, height, video_file=None, single_line=False, text_srt=None):
    """Bar hiding subtitles already burned into the source video.

    Returns the drawbox filters (or None), the MarginV and the FontSize to use
    for the translated subtitles. In "auto" mode the band is detected from the
    video itself; detection failures fall back to the fixed ratios in config.
    The rows it covers are fixed for the whole video, but its width follows the
    line under it -- see ``_measured_bars`` -- falling back to one bar as wide
    as the longest line when that cannot be measured. ``text_srt`` is the
    subtitle file actually being burned, when its timing differs from the source
    transcript's: the bar has to be up for both.
    """
    cfg = load_key_safe("cover_hardcoded_subtitles") or {}
    if not cfg.get("enable"):
        return None, TRANS_MARGIN_V, TRANS_FONT_SIZE

    y = bar_height = text_height = None
    bar_x, bar_width = 0, width
    if cfg.get("mode", "auto") == "auto" and video_file:
        from core.utils.hardsub_detect import detect_subtitle_band
        band = detect_subtitle_band(video_file, width, height, cfg.get("detection") or {})
        if band:
            y, bar_height = band["y"], band["height"]
            text_height = band.get("text_height")
            if cfg.get("limit_bar_width", True):
                bar_x, bar_width = band["x"], band["width"]
            rprint(f"[bold green]🔍 Detected hardcoded subtitles at y={y} "
                   f"({bar_height}px tall, {text_height}px of glyphs, "
                   f"confidence {band['confidence']}).[/bold green]")
        else:
            rprint("[bold yellow]⚠️ Auto-detection found no hardcoded subtitles; "
                   "falling back to the fixed ratios in config.[/bold yellow]")
    if y is None:
        y, bar_height = _fixed_bar(cfg, height)

    bar_color = cfg.get("color", "black")
    bar_opacity = float(cfg.get("opacity", 1.0))

    # The translated subtitles always render near the bottom of the frame, so a
    # bar detected in the upper half (subtitles burned at the top) must not drag
    # them up there -- only a bar in the lower half dictates their position.
    bar_in_lower_half = (y + bar_height / 2) > height / 2
    # A single line goes inside the bar, so it can be sized to the text it hides.
    text_in_bar = bar_in_lower_half and (single_line or not load_key_safe("burn_src_subtitles", True))

    font_size = TRANS_FONT_SIZE
    if text_in_bar and text_height and cfg.get("match_source_font_size", True):
        font_size = _matched_font_size(text_height, height, cfg)
        rprint(f"[bold green]🔠 Matching the translated font to the source text: "
               f"FontSize={font_size}.[/bold green]")

    drawbox = None
    # A bar that is up for the whole video cannot follow the line under it, so
    # the per-line widths only apply when the bar is allowed to switch off.
    if cfg.get("per_line_width", True) and cfg.get("time_gate", True) and video_file:
        drawbox = _measured_bars(video_file, y, bar_height, width, height, cfg, text_srt,
                                 font_size, text_in_bar, bar_color, bar_opacity)
    if drawbox is None:
        gate_srts = [SRC_SRT] + ([text_srt] if text_srt else [])
        enable_expr = _cover_enable_expr(gate_srts, cfg)
        drawbox = _drawbox(bar_x, y, bar_width, bar_height, bar_color, bar_opacity, enable_expr)
        rprint(f"[bold green]🩹 Covering hardcoded subtitles with a {bar_width}x{bar_height}px "
               f"{bar_color}@{bar_opacity} bar ({bar_height / height:.1%} of the frame height)"
               f"{' (only while a subtitle is on screen)' if enable_expr else ''}.[/bold green]")

    if not text_in_bar:
        # two stacked lines, or a bar elsewhere: keep the default bottom position
        return drawbox, TRANS_MARGIN_V, TRANS_FONT_SIZE

    center_ratio = (height - (y + bar_height / 2)) / height
    margin_v = max(2, round(center_ratio * PLAY_RES_Y - font_size / 2))
    return drawbox, margin_v, font_size

def merge_subtitles_to_video():
    from core._1_ytdlp import is_audio_only_input
    if is_audio_only_input():
        rprint("[bold green]🎵 Audio-only input: skipping video merge. Subtitle files are ready in the `output` directory.[/bold green]")
        return

    video_file = find_video_files()
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # Check resolution
    if not load_key("burn_subtitles"):
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    if not os.path.exists(SRC_SRT) or not os.path.exists(TRANS_SRT):
        rprint("Subtitle files not found in the 'output' directory.")
        exit(1)

    video = cv2.VideoCapture(video_file)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    cover_bar, trans_margin_v, trans_font_size = build_cover_bar(
        TARGET_WIDTH, TARGET_HEIGHT, video_file, text_srt=TRANS_SRT
    )

    filters = [
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease",
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
    ]
    # the bar goes down first so every subtitle below is drawn on top of it
    if cover_bar:
        filters.append(cover_bar)
    if load_key_safe("burn_src_subtitles", True):
        filters.append(
            f"subtitles={SRC_SRT}:force_style='FontSize={SRC_FONT_SIZE},FontName={FONT_NAME},"
            f"PrimaryColour={SRC_FONT_COLOR},OutlineColour={SRC_OUTLINE_COLOR},OutlineWidth={SRC_OUTLINE_WIDTH},"
            f"ShadowColour={SRC_SHADOW_COLOR},BorderStyle=1'"
        )
    filters.append(
        f"subtitles={TRANS_SRT}:force_style='FontSize={trans_font_size},FontName={TRANS_FONT_NAME},"
        f"PrimaryColour={TRANS_FONT_COLOR},OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
        f"Alignment=2,MarginV={trans_margin_v},{trans_backdrop_style(bool(cover_bar))}'"
    )

    ffmpeg_cmd = ['ffmpeg', '-i', video_file, '-vf', ",".join(filters).encode('utf-8')]
    ffmpeg_cmd.extend(video_encoder_args(video_file))
    ffmpeg_cmd.extend(audio_encoder_args(video_file))
    # moov at the front, so a multi-GB result seeks instantly instead of making
    # the player read the tail first.
    ffmpeg_cmd.extend(['-movflags', '+faststart', '-y', OUTPUT_VIDEO])

    rprint("🎬 Start merging subtitles to video...")
    start_time = time.time()
    process = subprocess.Popen(ffmpeg_cmd)

    try:
        process.wait()
        if process.returncode == 0:
            rprint(f"\n✅ Done! Time taken: {time.time() - start_time:.2f} seconds")
        else:
            rprint("\n❌ FFmpeg execution error")
    except Exception as e:
        rprint(f"\n❌ Error occurred: {e}")
        if process.poll() is None:
            process.kill()

if __name__ == "__main__":
    merge_subtitles_to_video()
