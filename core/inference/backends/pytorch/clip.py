from pathlib import Path
from typing import Any, Optional

from transformers import CLIPTextModel

from core.config import config
from ..base_backend import BaseCLIPBackend


class DynamicPyTorchCLIPBackend(BaseCLIPBackend):
    def __init__(self, directory: Optional[Path]) -> None:
        self.clip: CLIPTextModel
        self.directory = directory

    @property
    def backend_object(self) -> Any:
        return self.clip

    @backend_object.deleter
    def backend_object(self):
        self.clip = None  # type: ignore

    def load(self):
        if self.directory is not None:
            self.clip = CLIPTextModel.from_pretrained(self.directory / "text_encoder")
        else:
            self.clip = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.clip.to(config.api.device)

    def __call__(self, input_ids: Any, _: int = 77) -> Any:
        return self.clip(input_ids).text_embeddings
