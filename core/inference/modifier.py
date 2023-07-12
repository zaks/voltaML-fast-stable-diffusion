from abc import abstractmethod, ABC
from contextlib import ExitStack
from dataclasses import dataclass

import torch

from .pipeline import UnifiedPipeline


@dataclass
class DiffusionData:
    """Dataclass storing data about the current diffusion step."""

    pipeline: UnifiedPipeline

    time: int = 0
    width: int = 512
    height: int = 512
    cfg_scale: float = 1.0

    input_ids: torch.Tensor = None

    latents: torch.Tensor = None

    latent_model_input: torch.Tensor = None
    timesteps: torch.Tensor = None
    encoder_hidden_states: torch.Tensor = None

    noise_pred: torch.Tensor = None
    noise_pred_uncond: torch.Tensor = None
    noise_pred_text: torch.Tensor = None


class DiffusionModifier(ABC):
    """Abstract class for a modification step in the diffusion pipeline."""

    def __init__(
        self,
        display_name: str,
    ) -> None:
        self.display_name = display_name

    @abstractmethod
    def enter_context(self, context: ExitStack, data: DiffusionData) -> None:
        """Called when entering the UNet generation loop. This is executed ONCE and AFTER the text_encoder has finished running."""

    @abstractmethod
    def pre_unet(self, data: DiffusionData) -> DiffusionData:
        """Called before the UNet is run. This is executed EVERYTIME BEFORE the UNet is run."""
        return data

    @abstractmethod
    def post_unet(self, data: DiffusionData) -> DiffusionData:
        """Called after the UNet is run. This is executed EVERYTIME AFTER the UNet is run."""
        return data

    @abstractmethod
    def setup(self, context: ExitStack, data: DiffusionData) -> DiffusionData:
        """Called before the UNet is run. This is executed ONCE and BEFORE the text_encoder has finished running."""
        return data

    @abstractmethod
    def pre_vae(self, data: DiffusionData) -> DiffusionData:
        """Called before the VAE is run. This is executed BEFORE the VAE is run."""
        return data

    @abstractmethod
    def post_vae(self, data: DiffusionData) -> DiffusionData:
        """Called after the VAE is run. This is executed AFTER the VAE is run."""
        return data
