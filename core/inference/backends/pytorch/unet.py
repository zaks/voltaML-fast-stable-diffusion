from pathlib import Path
from typing import Any, List, Optional, Union

from diffusers import UNet2DConditionModel

from core.config import config
from ..base_backend import BaseUNetBackend


class DynamicPyTorchUNetBackend(BaseUNetBackend):
    def __init__(self, directory: Optional[Union[Path, str]]) -> None:
        self.unet: UNet2DConditionModel
        self.directory = directory

    def _no_controlnet(
        self,
        latent_model_input: Any,
        timesteps: Any,
        encoder_hidden_states: Any,
        width: int,
        height: int,
    ) -> Any:
        return self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    def _controlnet(
        self,
        latent_model_input: Any,
        timesteps: Any,
        encoder_hidden_states: Any,
        width: int,
        height: int,
        down_block_res_samples: List[Any],
        mid_block_res_sample: Any,
    ) -> Any:
        return self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

    @property
    def backend_object(self) -> Any:
        return self.unet

    @backend_object.deleter
    def backend_object(self):
        self.unet = None  # type: ignore

    @property
    def capabilities(self) -> BaseUNetBackend.UNetCapabilities:
        return BaseUNetBackend.UNetCapabilities(
            weighted_prompts=True,
            dynamic_sizes=True,
            controlnet=True,
            width=None,
            height=None,
        )

    def load(self):
        if self.directory is not None:
            if isinstance(self.directory, Path):
                self.unet = UNet2DConditionModel.from_pretrained(self.directory / "unet")  # type: ignore
            else:
                self.unet = UNet2DConditionModel.from_pretrained(self.directory)
        else:
            self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")  # type: ignore
        self.unet.to(config.api.device)
