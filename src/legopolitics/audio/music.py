INTERPRETIVE_MUSIC_LABELS = {"military-style music", "patriotic-style music"}


def is_interpretive_music_label(label: str) -> bool:
    return label.casefold() in INTERPRETIVE_MUSIC_LABELS
