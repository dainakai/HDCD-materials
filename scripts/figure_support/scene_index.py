"""Paper-facing scene indices for the fourteen 260715 recordings."""

from __future__ import annotations


SCENE_NUMBERS = (12, 13, 15, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)
SCENE_TO_INDEX = {
    raw_scene_number: paper_scene_index
    for paper_scene_index, raw_scene_number in enumerate(SCENE_NUMBERS, start=1)
}
INDEX_TO_SCENE = {
    paper_scene_index: raw_scene_number
    for raw_scene_number, paper_scene_index in SCENE_TO_INDEX.items()
}


def paper_scene_index(raw_scene_number: int) -> int:
    """Return the consecutive paper index (1--14) for a raw scene number."""

    return SCENE_TO_INDEX[int(raw_scene_number)]


def raw_scene_name(raw_scene_number: int) -> str:
    """Return the acquisition-system scene identifier."""

    return f"C000H001S{int(raw_scene_number):04d}"


def paper_scene_label(raw_scene_number: int) -> str:
    """Return the short scene label used in paper figures."""

    return f"Scene {paper_scene_index(raw_scene_number)}"
