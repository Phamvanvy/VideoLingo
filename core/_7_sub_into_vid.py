import os, subprocess, time
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


def build_cover_bar(width, height, video_file=None, single_line=False):
    """Black bar hiding subtitles already burned into the source video.

    Returns the drawbox filter (or None), the MarginV and the FontSize to use
    for the translated subtitles. In "auto" mode the band is detected from the
    video itself; detection failures fall back to the fixed ratios in config.
    """
    cfg = load_key_safe("cover_hardcoded_subtitles") or {}
    if not cfg.get("enable"):
        return None, TRANS_MARGIN_V, TRANS_FONT_SIZE

    y = bar_height = text_height = None
    if cfg.get("mode", "auto") == "auto" and video_file:
        from core.utils.hardsub_detect import detect_subtitle_band
        band = detect_subtitle_band(video_file, height, cfg.get("detection") or {})
        if band:
            y, bar_height = band["y"], band["height"]
            text_height = band.get("text_height")
            rprint(f"[bold green]🔍 Detected hardcoded subtitles at y={y} "
                   f"({bar_height}px tall, {text_height}px of glyphs, "
                   f"confidence {band['confidence']}).[/bold green]")
        else:
            rprint("[bold yellow]⚠️ Auto-detection found no hardcoded subtitles; "
                   "falling back to the fixed ratios in config.[/bold yellow]")
    if y is None:
        y, bar_height = _fixed_bar(cfg, height)

    drawbox = f"drawbox=x=0:y={y}:w={width}:h={bar_height}:color=black@1.0:t=fill"
    rprint(f"[bold green]🩹 Covering hardcoded subtitles with a {bar_height}px black bar "
           f"({bar_height / height:.1%} of the frame).[/bold green]")

    # The translated subtitles always render near the bottom of the frame, so a
    # bar detected in the upper half (subtitles burned at the top) must not drag
    # them up there -- only a bar in the lower half dictates their position.
    bar_in_lower_half = (y + bar_height / 2) > height / 2
    if not bar_in_lower_half or not (single_line or not load_key_safe("burn_src_subtitles", True)):
        # two stacked lines, or a bar elsewhere: keep the default bottom position
        return drawbox, TRANS_MARGIN_V, TRANS_FONT_SIZE

    # A single line goes inside the bar, so it can be sized to the text it hides.
    font_size = TRANS_FONT_SIZE
    if text_height and cfg.get("match_source_font_size", True):
        font_size = _matched_font_size(text_height, height, cfg)
        rprint(f"[bold green]🔠 Matching the translated font to the source text: "
               f"FontSize={font_size}.[/bold green]")
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
    cover_bar, trans_margin_v, trans_font_size = build_cover_bar(TARGET_WIDTH, TARGET_HEIGHT, video_file)

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

    ffmpeg_gpu = load_key("ffmpeg_gpu")
    if ffmpeg_gpu:
        rprint("[bold green]will use GPU acceleration.[/bold green]")
        ffmpeg_cmd.extend(['-c:v', 'h264_nvenc'])
    ffmpeg_cmd.extend(['-y', OUTPUT_VIDEO])

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
