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

def build_cover_bar(width, height):
    """Black bar hiding subtitles already burned into the source video.

    Returns the drawbox filter (or None) and the MarginV to use for the
    translated subtitles so they land on top of that bar.
    """
    cfg = load_key_safe("cover_hardcoded_subtitles") or {}
    if not cfg.get("enable"):
        return None, TRANS_MARGIN_V

    height_ratio = float(cfg.get("height_ratio", 0.16))
    offset_ratio = float(cfg.get("bottom_offset_ratio", 0.0))
    bar_height = max(1, int(height * height_ratio))
    offset = int(height * offset_ratio)
    y = max(0, height - bar_height - offset)
    drawbox = f"drawbox=x=0:y={y}:w={width}:h={bar_height}:color=black@1.0:t=fill"

    if load_key_safe("burn_src_subtitles", True):
        # source line sits at the default MarginV, translation stacks above it
        margin_v = TRANS_MARGIN_V
    else:
        # single line, center it vertically inside the bar
        center_ratio = offset_ratio + height_ratio / 2
        margin_v = max(2, round(center_ratio * PLAY_RES_Y - TRANS_FONT_SIZE / 2))

    rprint(f"[bold green]🩹 Covering hardcoded subtitles with a {bar_height}px black bar.[/bold green]")
    return drawbox, margin_v

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
    cover_bar, trans_margin_v = build_cover_bar(TARGET_WIDTH, TARGET_HEIGHT)

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
        f"subtitles={TRANS_SRT}:force_style='FontSize={TRANS_FONT_SIZE},FontName={TRANS_FONT_NAME},"
        f"PrimaryColour={TRANS_FONT_COLOR},OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
        f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV={trans_margin_v},BorderStyle=4'"
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
