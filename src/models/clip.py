"""A single editable segment in a video project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class Clip:
    id: str = ""
    index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    enabled: bool = True
    part_text: str = ""
    source_video: str = ""
    subtitle_file: str = ""
    preview_path: str = ""
    export_path: str = ""
    voiceover_script: str = ""
    custom_header: str = ""
    custom_subtitle: str = ""
    is_review_master: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Preserve fields added dynamically by newer UI code.
        data.update({k: v for k, v in vars(self).items() if k not in data})
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Clip":
        known = {field.name for field in fields(cls)}
        obj = cls(**{key: value for key, value in data.items() if key in known})
        for key, value in data.items():
            if key not in known:
                setattr(obj, key, value)
        return obj
