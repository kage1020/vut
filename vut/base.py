from vut.mapping import (
    load_action_mapping,
    load_class_mapping,
    load_video_action_mapping,
    load_video_boundaries,
)
from vut.util import Env


class Base:
    env = Env()
    text_to_index: dict[str, int] = {}
    index_to_text: dict[int, str] = {}
    action_to_steps: dict[str, list[str]] = {}
    video_to_action: dict[str, str] = {}
    video_boundaries: dict[str, list[tuple[int, int]]] = {}
    backgrounds: list[str] = []

    def __init__(
        self,
        class_mapping_path: str | None = None,
        class_mapping_has_header: bool = None,
        class_mapping_separator: str | None = ",",
        action_mapping_path: str | None = None,
        action_mapping_has_header: bool = False,
        action_mapping_action_separator: str = ",",
        action_mapping_step_separator: str = " ",
        video_action_mapping_path: str | None = None,
        video_action_mapping_has_header: bool = False,
        video_action_mapping_separator: str = ",",
        video_boundary_dir_path: str | None = None,
        video_boundary_has_header: bool = False,
        video_boundary_separator: str = ",",
        backgrounds: list[str] | None = None,
    ):
        if class_mapping_path is not None:
            text_to_index, index_to_text = load_class_mapping(
                class_mapping_path,
                has_header=class_mapping_has_header,
                separator=class_mapping_separator,
            )
            self.text_to_index = text_to_index
            self.index_to_text = index_to_text
        if action_mapping_path is not None:
            action_to_steps = load_action_mapping(
                action_mapping_path,
                has_header=action_mapping_has_header,
                action_separator=action_mapping_action_separator,
                step_separator=action_mapping_step_separator,
            )
            self.action_to_steps = action_to_steps
        if video_action_mapping_path is not None:
            video_to_action = load_video_action_mapping(
                video_action_mapping_path,
                has_header=video_action_mapping_has_header,
                separator=video_action_mapping_separator,
            )
            self.video_to_action = video_to_action
        if video_boundary_dir_path is not None:
            video_boundaries = load_video_boundaries(
                video_boundary_dir_path,
                has_header=video_boundary_has_header,
                separator=video_boundary_separator,
            )
            self.video_boundaries = video_boundaries
        if backgrounds is not None:
            self.backgrounds = backgrounds
