"""Project and export configuration models with JSON compatibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .clip import Clip


PLATFORMS = ["TikTok", "YouTube Shorts", "Facebook Reels", "Instagram Reels"]
ASPECT_MODES = {
    "center_crop": "Cắt giữa (lấp đầy)",
    "keep_ratio": "Giữ tỷ lệ (nền đen)",
    "blur_bg": "Giữ tỷ lệ (nền mờ)",
    "image_bg": "Nền ảnh",
    "custom_crop": "Cắt tùy chỉnh",
}
PART_TEXT_MODES = {
    "overlay": "Chữ phủ",
    "outline": "Chữ viền",
    "lower_third": "Thanh tiêu đề",
    "shrink": "Thu video, chừa đầu trang",
}
SUBTITLE_STYLES = {
    "none": "Không dùng",
    "outline": "Chữ viền",
    "karaoke": "Karaoke",
    "bar": "Thanh phụ đề",
    "shrink_bar": "Thu video + thanh phụ đề",
}


@dataclass
class ExportConfig:
    platform: str = "TikTok"
    width: int = 1080
    height: int = 1920
    aspect_mode: str = "center_crop"
    part_text_enabled: bool = True
    part_text_mode: str = "overlay"
    part_text_position: str = "top"
    part_text_font: str = "Arial"
    part_text_bold: bool = True
    part_text_italic: bool = False
    part_text_underline: bool = False
    part_text_fontsize: int = 72
    part_text_color: str = "white"
    part_text_bg_color: str = "black"
    part_text_y_pct: float = -1.0
    header_height: int = 250
    watermark_text: str = ""
    watermark_opacity: float = 0.35
    crop_x: float = 0.5
    crop_y: float = 0.5
    crop_w: float = 1.0
    crop_h: float = 1.0
    pre_crop_enabled: bool = False
    pre_crop_x: float = 0.5
    pre_crop_y: float = 0.5
    pre_crop_w: float = 1.0
    pre_crop_h: float = 1.0
    background_image: str = ""
    background_music_enabled: bool = False
    background_music_path: str = ""
    background_music_volume: int = 15
    remove_watermark_enabled: bool = False
    remove_watermark_mode: str = "blur"
    remove_watermark_x: float = 0.82
    remove_watermark_y: float = 0.88
    remove_watermark_w: float = 0.16
    remove_watermark_h: float = 0.08
    remove_watermark_strength: int = 3
    remove_watermark_color: str = "black"
    remove_watermark_regions: list[dict[str, float]] = field(default_factory=list)
    use_gpu: bool = False
    preset: str = "fast"
    crf: int = 23
    audio_bitrate: str = "192k"
    subtitle_enabled: bool = False
    global_subtitle_file: str = ""
    subtitle_style: str = "none"
    subtitle_font: str = "Arial"
    subtitle_bold: bool = True
    subtitle_italic: bool = False
    subtitle_underline: bool = False
    subtitle_fontsize: int = 65
    subtitle_color: str = "white"
    subtitle_highlight_color: str = "yellow"
    subtitle_position: str = "bottom"
    subtitle_margin_v: int = 0
    subtitle_y_pct: float = -1.0
    subtitle_bg_color: str = "white"
    subtitle_bar_height: int = 200

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update({k: v for k, v in vars(self).items() if k not in data})
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExportConfig":
        data = data or {}
        known = {item.name for item in fields(cls)}
        obj = cls(**{key: value for key, value in data.items() if key in known})
        for key, value in data.items():
            if key not in known:
                setattr(obj, key, value)
        return obj


@dataclass
class Project:
    id: str = ""
    name: str = ""
    created_at: str = ""
    updated_at: str = ""
    source_video: str = ""
    original_source_video: str = ""
    video_metadata: dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""
    clips: list[Clip] = field(default_factory=list)
    export_config: ExportConfig = field(default_factory=ExportConfig)
    clip_duration: float = 60.0
    transcript_file: str = ""
    source_transcript_file: str = ""
    narration_transcript_file: str = ""
    narration_subtitle_file: str = ""
    transcript_language: str = ""
    voiceover_script: str = ""
    voiceover_audio: str = ""
    review_base_video: str = ""
    review_voice_video: str = ""
    final_video: str = ""
    global_output_dir: str = ""
    workflow_mode: str = "review"
    workflow_stage: str = "source"
    scene_revision: int = 0
    script_scene_revision: int = -1
    voice_scene_revision: int = -1
    subtitle_voice_revision: int = -1
    review_voice_duration: float = 0.0
    review_video_duration: float = 0.0
    review_sync_error: float = 0.0
    voice_alignment_ok: bool = False
    voice_rate_min: float = 1.10
    voice_rate_max: float = 1.15
    ai_script_extra_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["clips"] = [clip.to_dict() for clip in self.clips]
        data["export_config"] = self.export_config.to_dict()
        data.update({k: v for k, v in vars(self).items() if k not in data})
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        known = {item.name for item in fields(cls)}
        values = {key: value for key, value in data.items() if key in known}
        values["clips"] = [
            item if isinstance(item, Clip) else Clip.from_dict(item)
            for item in (data.get("clips") or [])
        ]
        values["export_config"] = ExportConfig.from_dict(data.get("export_config"))
        obj = cls(**values)
        if not obj.original_source_video:
            obj.original_source_video = obj.source_video
        for key, value in data.items():
            if key not in known:
                setattr(obj, key, value)
        return obj
