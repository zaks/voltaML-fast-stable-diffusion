from pathlib import Path
from typing import Any, Optional, Union

from diffusers import AutoencoderKL

from core.config import config
from ..base_backend import BaseVAEBackend


class DynamicPyTorchVAEBackend(BaseVAEBackend):
    def __init__(
        self, directory: Optional[Union[Path, str]], vae: Optional[AutoencoderKL]
    ) -> None:
        self.vae: AutoencoderKL
        self.directory = directory
        if vae is not None:
            self.vae = vae

    def load(self):
        if self.vae is None:
            if self.directory is not None:
                if isinstance(self.directory, Path):
                    self.vae = AutoencoderKL.from_pretrained(self.directory / "vae")  # type: ignore
                else:
                    self.vae = AutoencoderKL.from_pretrained(self.directory)  # type: ignore
            else:
                self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-5")  # type: ignore
        self.vae.to(config.api.device)

    @property
    def capabilities(self) -> BaseVAEBackend.VAECapabilities:
        return BaseVAEBackend.VAECapabilities(encode=True, decode=True)

    @property
    def backend_object(self) -> Any:
        return self.vae

    @backend_object.deleter
    def backend_object(self):
        self.vae = None  # type: ignore

    def encode(self, latents: Any, width: int, height: int) -> Any:
        return self.vae.encode(latents).latent_dist  # type: ignore

    def decode(self, latents: Any, width: int, height: int) -> Any:
        return self.vae.decode(latents).sample  # type: ignore
