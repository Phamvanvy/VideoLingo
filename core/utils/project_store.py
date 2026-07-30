"""Reload a project that was archived to ``history/``.

``cleanup`` moves a finished ``output/`` into ``history/<video name>/``; this
module is the way back. Everything the pipeline produces up to the translation
is text -- chunks, terminology, the translation spreadsheets, the .srt files --
so an archive is small once the burnt-in ``output_sub.mp4`` is left out of it.
Restoring one puts those artifacts back where every stage expects them, and
"Burn subtitles into the video only" in the re-run panel then rebuilds the video
from ``src.srt`` and ``trans.srt`` alone: no Whisper, no LLM.
"""

import os
import shutil

from core.utils.models import _OUTPUT_DIR, _TEXT_DONE_MARKER

HISTORY_DIR = "history"

# Rebuilt from the subtitles by one ffmpeg pass, and by far the largest files in
# a project, so archiving them roughly doubles a project for nothing.
RENDERED_OUTPUTS = ("output_sub.mp4", "output_dub.mp4")

# The minimum an archive needs before "burn subtitles only" can run against it.
REQUIRED_ARTIFACTS = ("src.srt", "trans.srt")


def _allowed_formats():
    from core.utils.config_utils import load_key

    return set(load_key("allowed_video_formats")), set(load_key("allowed_audio_formats"))


def find_project_media(project_dir):
    """Source media inside an archived project, as ``(path, type)``.

    Deliberately not ``_1_ytdlp.find_media_file``: that one resolves the
    manifest against ``output/`` and only filters rendered videos by the literal
    prefix ``output/output``, so it counts a ``history/x/output_sub.mp4`` as a
    second input video and refuses the folder.
    """
    if not os.path.isdir(project_dir):
        return None, None

    from core._1_ytdlp import GENERATED_AUDIO_NAMES

    video_formats, audio_formats = _allowed_formats()
    videos, audios = [], []
    for entry in sorted(os.listdir(project_dir)):
        path = os.path.join(project_dir, entry).replace("\\", "/")
        if entry in RENDERED_OUTPUTS or not os.path.isfile(path):
            continue
        ext = os.path.splitext(entry)[1][1:].lower()
        if ext in video_formats:
            videos.append(path)
        elif ext in audio_formats and entry not in GENERATED_AUDIO_NAMES:
            audios.append(path)

    if videos:
        return videos[0], "video"
    if audios:
        return audios[0], "audio"
    return None, None


def list_projects(history_dir=HISTORY_DIR):
    """Names of every archived project, newest first."""
    if not os.path.isdir(history_dir):
        return []
    names = [
        name
        for name in os.listdir(history_dir)
        if os.path.isdir(os.path.join(history_dir, name))
    ]
    names.sort(key=lambda n: os.path.getmtime(os.path.join(history_dir, n)), reverse=True)
    return names


def project_status(name, history_dir=HISTORY_DIR):
    """What an archived project holds and whether it can be restored."""
    project_dir = os.path.join(history_dir, name)
    media, media_type = find_project_media(project_dir)
    missing = [
        artifact
        for artifact in REQUIRED_ARTIFACTS
        if not os.path.exists(os.path.join(project_dir, artifact))
    ]
    return {
        "name": name,
        "dir": project_dir,
        "media": media,
        "media_type": media_type,
        "missing": missing,
        "has_rendered": any(
            os.path.exists(os.path.join(project_dir, rendered))
            for rendered in RENDERED_OUTPUTS
        ),
        "restorable": bool(media) and not missing,
    }


def restore_project(name, history_dir=HISTORY_DIR, output_dir=_OUTPUT_DIR):
    """Copy an archived project back into ``output/``, ready for a re-burn.

    Copies rather than moves: a re-burn that goes wrong should not cost the
    archive. Returns the project status that was restored.
    """
    from core._1_ytdlp import write_input_manifest

    status = project_status(name, history_dir)
    if not status["media"]:
        raise ValueError(f"No source media file found in {status['dir']}")
    if status["missing"]:
        raise ValueError(
            f"{status['dir']} is missing {', '.join(status['missing'])}"
        )

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(status["dir"], output_dir)

    # ``cleanup`` sanitizes file names on the way into the archive, so the
    # archived manifest can name a file that no longer exists. Rewrite it
    # against what actually landed in ``output/``.
    media_name = os.path.basename(status["media"])
    write_input_manifest(f"{output_dir}/{media_name}", status["media_type"], output_dir)

    # ``cleanup`` archives via ``glob``, which skips dotfiles, so the done
    # marker never made it into the archive. Rewrite it: the artifacts checked
    # above are exactly what it asserts, which is what lets the UI open on the
    # re-run panel instead of offering to transcribe the video again.
    marker = os.path.join(output_dir, os.path.basename(_TEXT_DONE_MARKER))
    with open(marker, "w", encoding="utf-8") as f:
        f.write("")

    return status
