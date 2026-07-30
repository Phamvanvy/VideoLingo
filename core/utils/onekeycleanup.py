import os
import glob
from core._1_ytdlp import find_media_file
from core.utils.models import _TEXT_DONE_MARKER, _AUDIO_DONE_MARKER
from core.utils.project_store import RENDERED_OUTPUTS
import shutil

def cleanup(history_dir="history", keep_rendered=True):
    """Move ``output/`` into ``history/<video name>/``.

    With ``keep_rendered=False`` the burnt-in videos are dropped instead of
    archived. They are the largest files by far and one "burn subtitles into the
    video only" re-run rebuilds them from the .srt files that *are* archived, so
    leaving them out is what makes an archive cheap enough to keep around.
    """
    # Get input media file name
    media_file, _ = find_media_file()
    video_name = os.path.splitext(os.path.basename(media_file))[0]
    video_name = sanitize_filename(video_name)

    # Create required folders
    os.makedirs(history_dir, exist_ok=True)
    video_history_dir = os.path.join(history_dir, video_name)
    log_dir = os.path.join(video_history_dir, "log")
    gpt_log_dir = os.path.join(video_history_dir, "gpt_log")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(gpt_log_dir, exist_ok=True)

    # Move non-log files
    for file in glob.glob("output/*"):
        if os.path.basename(file) in RENDERED_OUTPUTS and not keep_rendered:
            # Deleted rather than left behind, so the empty-output rmdir below
            # still succeeds and the next project starts from a clean slate.
            try:
                os.remove(file)
                print(f"🗑️ Dropped rendered video: {file}")
            except OSError as e:
                print(f"⚠️ Could not delete {file}: {e}")
            continue
        if not file.endswith(('log', 'gpt_log')):
            move_file(file, video_history_dir)

    # Move log files
    for file in glob.glob("output/log/*"):
        move_file(file, log_dir)

    # Move gpt_log files
    for file in glob.glob("output/gpt_log/*"):
        move_file(file, gpt_log_dir)

    # Done markers are dotfiles, which the globs above skip. Delete them rather
    # than archive them: leaving one behind keeps "output" alive and makes the
    # next video look like its subtitles were already generated.
    for marker in (_TEXT_DONE_MARKER, _AUDIO_DONE_MARKER):
        try:
            os.remove(marker)
        except OSError:
            pass

    # Delete empty output directories
    try:
        os.rmdir("output/log")
        os.rmdir("output/gpt_log")
        os.rmdir("output")
    except OSError:
        pass  # Ignore errors when deleting directories

    return video_history_dir

def move_file(src, dst):
    try:
        # Get the source file name
        src_filename = os.path.basename(src)
        # Use os.path.join to ensure correct path and include file name
        dst = os.path.join(dst, sanitize_filename(src_filename))
        
        if os.path.exists(dst):
            if os.path.isdir(dst):
                # If destination is a folder, try to delete its contents
                shutil.rmtree(dst, ignore_errors=True)
            else:
                # If destination is a file, try to delete it
                os.remove(dst)
        
        shutil.move(src, dst, copy_function=shutil.copy2)
        print(f"✅ Moved: {src} -> {dst}")
    except PermissionError:
        print(f"⚠️ Permission error: Cannot delete {dst}, attempting to overwrite")
        try:
            shutil.copy2(src, dst)
            os.remove(src)
            print(f"✅ Copied and deleted source file: {src} -> {dst}")
        except Exception as e:
            print(f"❌ Move failed: {src} -> {dst}")
            print(f"Error message: {str(e)}")
    except Exception as e:
        print(f"❌ Move failed: {src} -> {dst}")
        print(f"Error message: {str(e)}")

def sanitize_filename(filename):
    # Remove or replace disallowed characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

if __name__ == "__main__":
    cleanup()
