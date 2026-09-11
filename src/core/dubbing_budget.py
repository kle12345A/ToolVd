"""Editable, cue-based word budgets; measurements are at native TTS speed."""
import math


def group_dubbing_segments(segments):
    """Join adjacent fragments without changing words or crossing long pauses."""
    groups = []
    current = None
    for row in segments:
        if current is not None:
            span = current['end'] - current['start']
            sentence_end = current['text_vi'].rstrip('”\"\u2019\' ').endswith(('.', '!', '?', '。'))
            if (row['start'] - current['end'] >= 0.5
                    or row['end'] - current['start'] > 8.0
                    or (span >= 4.0 and sentence_end)):
                groups.append(current)
                current = None
        if current is None:
            current = dict(start=row['start'], end=row['end'], text_vi=row['text_vi'])
        else:
            current['end'] = row['end']
            current['text_vi'] += ' ' + row['text_vi']
    if current is not None:
        groups.append(current)
    return groups


def word_budget(segments, rate, measurements=None):
    segments = group_dubbing_segments(segments)
    rate = max(1.10, min(1.15, float(rate)))
    measurements = measurements or []
    count = sum(row.get("words", 0) for row in measurements)
    seconds = sum(row.get("seconds", 0) for row in measurements)
    # Conservative initial estimate; measured voice replaces this after synthesis.
    seconds_per_word = seconds / count if count and seconds > 0 else 1 / 3.5
    rows = []
    for index, segment in enumerate(segments):
        words = len(segment["text_vi"].split())
        slot = max(0.1, segment["end"] - segment["start"])
        measured = next((m for m in measurements if m.get("text") == segment["text_vi"]), None)
        native = measured["seconds"] if measured else words * seconds_per_word
        local_seconds_per_word = native / words if words else seconds_per_word
        target = max(0, math.floor(slot * rate / local_seconds_per_word))
        rows.append(dict(start=segment["start"], words=words, target=target,
                         remove=max(0, words-target), add=max(0, target-words),
                         duration=native/rate))
    return rows


def timeline_overflow(rows, video_duration):
    """Use the same sequential cue placement as the audio renderer."""
    finish = 0.0
    for row in rows:
        finish = max(finish, row['start']) + row['duration']
    return max(0.0, finish - video_duration)
