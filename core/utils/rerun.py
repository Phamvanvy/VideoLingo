"""Re-run the subtitle pipeline from a chosen stage, keeping earlier work.

Every stage already skips itself when its output file exists (see
``check_file_exists``), so "re-run from here" is really "delete the outputs of
this stage and everything after it". Swapping the translation model or the way
the subtitles are drawn then costs one stage instead of a full transcription.

The GPT cache in ``output/gpt_log`` needs the same treatment and is easy to
miss: ``ask_gpt`` keys its cache on the prompt alone, so a stage re-run against
a different model replays the *old* model's answers unless the matching log is
dropped too. Each stage therefore lists its gpt_log files alongside its outputs.
"""

import glob
import os

from core.utils.models import (
    _2_CLEANED_CHUNKS,
    _3_1_SPLIT_BY_NLP,
    _3_2_SPLIT_BY_MEANING,
    _4_1_TERMINOLOGY,
    _4_2_TRANSLATION,
    _5_SPLIT_SUB,
    _5_REMERGED,
    _8_1_AUDIO_TASK,
    _AUDIO_DIR,
    _AUDIO_SEGS_DIR,
    _OUTPUT_DIR,
    _TEXT_DONE_MARKER,
    _AUDIO_DONE_MARKER,
)

GPT_LOG_DIR = "output/gpt_log"
SUB_VIDEO = "output/output_sub.mp4"

# Ordered as the pipeline runs; the index doubles as the offset into the step
# list that st.py builds, so stage N re-runs steps[N:].
STAGES = [
    {
        "key": "asr",
        "label": "Transcription (Whisper) and everything after",
        "artifacts": [_2_CLEANED_CHUNKS],
    },
    {
        "key": "split",
        "label": "Sentence segmentation and everything after",
        "artifacts": [_3_1_SPLIT_BY_NLP, _3_2_SPLIT_BY_MEANING],
        "gpt_logs": ["split_by_meaning"],
    },
    {
        "key": "translate",
        "label": "Translation and everything after (use for a new model)",
        "artifacts": [_4_1_TERMINOLOGY, _4_2_TRANSLATION],
        "gpt_logs": ["summary", "translate_*"],
    },
    {
        "key": "subs",
        "label": "Subtitle splitting and alignment, then re-burn",
        "artifacts": [
            _5_SPLIT_SUB,
            _5_REMERGED,
            f"{_OUTPUT_DIR}/src.srt",
            f"{_OUTPUT_DIR}/trans.srt",
            f"{_OUTPUT_DIR}/src_trans.srt",
            f"{_OUTPUT_DIR}/trans_src.srt",
            f"{_AUDIO_DIR}/src_subs_for_audio.srt",
            f"{_AUDIO_DIR}/trans_subs_for_audio.srt",
        ],
        "gpt_logs": ["align_subs"],
    },
    {
        "key": "burn",
        "label": "Burn subtitles into the video only (no LLM, no re-translation)",
        "artifacts": [SUB_VIDEO],
    },
]

STAGE_KEYS = [stage["key"] for stage in STAGES]

# Rebuilt from the subtitles, so anything that changes an .srt invalidates them.
_DUBBING_ARTIFACTS = [
    _8_1_AUDIO_TASK,
    _AUDIO_SEGS_DIR,
    f"{_OUTPUT_DIR}/dub.srt",
    f"{_OUTPUT_DIR}/dub.mp3",
    f"{_OUTPUT_DIR}/dub.wav",
    f"{_OUTPUT_DIR}/output_dub.mp4",
    _AUDIO_DONE_MARKER,
]
_DUBBING_GPT_LOGS = ["sub_trim"]


def stage_index(key):
    """Position of a stage in the pipeline, for slicing the step list."""
    return STAGE_KEYS.index(key)


def _remove(path):
    try:
        if os.path.isdir(path):
            import shutil

            shutil.rmtree(path, ignore_errors=True)
            return True
        if os.path.exists(path):
            os.remove(path)
            return True
    except OSError:
        pass  # a file held open by a player is not worth failing the re-run over
    return False


def _paths_for(stage):
    for pattern in stage.get("gpt_logs", []):
        yield from glob.glob(os.path.join(GPT_LOG_DIR, f"{pattern}.json"))
    yield from stage["artifacts"]


def invalidate_from(key, drop_dubbing=None):
    """Delete the outputs of ``key`` and every later stage.

    ``drop_dubbing`` defaults to True for every stage but "burn", because those
    stages rewrite the subtitles the dubbing was built from. Returns the list of
    paths actually removed.
    """
    start = stage_index(key)
    removed = []

    for stage in STAGES[start:]:
        removed += [path for path in _paths_for(stage) if _remove(path)]

    if drop_dubbing is None:
        drop_dubbing = key != "burn"
    if drop_dubbing:
        for pattern in _DUBBING_GPT_LOGS:
            removed += [p for p in glob.glob(os.path.join(GPT_LOG_DIR, f"{pattern}.json")) if _remove(p)]
        removed += [path for path in _DUBBING_ARTIFACTS if _remove(path)]

    # Always, so the UI stops treating the subtitle stage as finished.
    if _remove(_TEXT_DONE_MARKER):
        removed.append(_TEXT_DONE_MARKER)
    return removed
