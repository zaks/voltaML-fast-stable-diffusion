from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, List

from core.config import config


class BaseVAEBackend(ABC):
    """Abstract class for VAE (encode and decode) functions per backend."""

    @dataclass
    class VAECapabilities:
        """Dataclass containing information about what this VAE can do."""

        encode: bool
        decode: bool

    @property
    @abstractmethod
    def capabilities(self) -> VAECapabilities:
        """Get the capabilities for this backend"""
        return BaseVAEBackend.VAECapabilities(False, False)

    @property
    @abstractmethod
    def backend_object(self) -> Any:
        """Object that gets called upon processing. Needed for some optimizations/modifications."""

    @backend_object.deleter
    @abstractmethod
    def backend_object(self):
        """Deleter for backend_object."""

    @abstractmethod
    def load(self):
        """Load this object into memory (on device)"""

    def unload(self):
        """Unload this object from memory"""
        if config.api.unload_destination == "null":
            del self.backend_object
        elif config.api.unload_destination == "cpu":
            self.backend_object.to("cpu")

    @abstractmethod
    def encode(self, latents: Any, width: int, height: int) -> Any:
        """Encode image"""

    @abstractmethod
    def decode(self, latents: Any, width: int, height: int) -> Any:
        """Decode image"""


class BaseUNetBackend(ABC):
    """Abstract class for UNet functions per backend."""

    @dataclass
    class UNetCapabilities:
        """Dataclass containing information about what this UNet can do."""

        weighted_prompts: bool
        dynamic_sizes: bool
        controlnet: bool
        width: Optional[int] = 512
        height: Optional[int] = 512

    @property
    @abstractmethod
    def capabilities(self) -> UNetCapabilities:
        """Get the capabilities for this backend"""
        return BaseUNetBackend.UNetCapabilities(True, True, True, None, None)

    @property
    @abstractmethod
    def backend_object(self) -> Any:
        """Object that gets called upon processing. Needed for some optimizations/modifications."""

    @backend_object.deleter
    @abstractmethod
    def backend_object(self):
        """Deleter for backend_object."""

    @abstractmethod
    def load(self):
        """Load this object into memory (on device)"""

    def unload(self):
        """Unload this object from memory"""
        if config.api.unload_destination == "null":
            del self.backend_object
        elif config.api.unload_destination == "cpu":
            self.backend_object.to("cpu")

    @abstractmethod
    def _no_controlnet(
        self,
        latent_model_input: Any,
        timesteps: Any,
        encoder_hidden_states: Any,
        width: int,
        height: int,
    ) -> Any:
        """Execute a step without controlnet samples."""

    @abstractmethod
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
        """Execute a step with controlnet samples."""

    def __call__(
        self,
        latent_model_input: Any,
        timesteps: Any,
        encoder_hidden_states: Any,
        width: int,
        height: int,
        down_block_res_samples: Optional[List[Any]] = None,
        mid_block_res_sample: Optional[Any] = None,
    ) -> Any:
        """Execute a step"""
        if down_block_res_samples is None or mid_block_res_sample is None:
            return self._no_controlnet(
                latent_model_input, timesteps, encoder_hidden_states, width, height
            )
        else:
            return self._controlnet(
                latent_model_input,
                timesteps,
                encoder_hidden_states,
                width,
                height,
                down_block_res_samples,
                mid_block_res_sample,
            )


class BaseCLIPBackend(ABC):
    """Abstract class for UNet functions per backend."""

    @property
    @abstractmethod
    def backend_object(self) -> Any:
        """Object that gets called upon processing. Needed for some optimizations/modifications."""

    @backend_object.deleter
    @abstractmethod
    def backend_object(self):
        """Deleter for backend_object."""

    @abstractmethod
    def load(self):
        """Load this object into memory (on device)"""

    def unload(self):
        """Unload this object from memory"""
        if config.api.unload_destination == "null":
            del self.backend_object
        elif config.api.unload_destination == "cpu":
            self.backend_object.to("cpu")

    @abstractmethod
    def __call__(
        self,
        input_ids: Any,
        sequence_length: int = 77,
    ) -> Any:
        """Execute a step"""
