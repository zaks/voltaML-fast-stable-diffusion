from pathlib import Path
from typing import Any, Union

from aitemplate.compiler import Model
import torch

from core.aitemplate.src.compile_lib_dynamic.compile_vae_alt import (
    ait_AutoencoderKL,
    map_vae_params,
)
from ...base_backend import BaseVAEBackend
from ..common import init_ait_module, nchw, nhwc


class DynamicAITemplateVAEBackend(BaseVAEBackend):
    def __init__(self, directory: Union[Path, str]) -> None:
        self.model: Model
        self.directory = directory

    def load(self):
        vae_pt = None  # TODO: load in pt vae
        ait_vae = ait_AutoencoderKL(
            1,
            64,
            64,
            in_channels=3,
            out_channels=3,
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],  # type: ignore
            up_block_types=[
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],  # type: ignore
            block_out_channels=[128, 256, 512, 512],  # type: ignore
            layers_per_block=2,
            act_fn="silu",
            latent_channels=4,
            sample_size=512,
        )
        state_dict = map_vae_params(ait_vae, vae_pt)
        self.model = init_ait_module("AutoencoderKL", str(self.directory), state_dict)
        del state_dict

    def unload(self):
        del self.backend_object

    @property
    def backend_object(self) -> Any:
        return self.model

    @backend_object.deleter
    def backend_object(self):
        self.model = None  # type: ignore

    def decode(self, latents: Any, width: int, height: int) -> Any:
        inputs = [nhwc(latents)]
        ys = []
        num_outputs = len(self.model.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = self.model.get_output_maximum_shape(i)
            shape[0] = 1  # self.batch
            shape[1] = height
            shape[2] = width
            ys.append(torch.empty(shape).cuda().half())
        self.model.run_with_tensors(inputs, ys, graph_mode=False)
        vae_out = nchw(ys[0])
        return vae_out
