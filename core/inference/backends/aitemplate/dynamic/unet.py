from pathlib import Path
from typing import Any, Dict, Union

from aitemplate.compiler import Model
import torch

from ...base_backend import BaseUNetBackend
from ...pytorch import DynamicPyTorchUNetBackend
from ..common import init_ait_module, nchw, nhwc


class DynamicAITemplateUNetBackend(BaseUNetBackend):
    def __init__(self, directory: Path) -> None:
        self.model: Model
        self.directory = directory

    def load(self):
        md_name = self.directory.stem.split("-")[0]
        # TODO: change this to resolve diffusers ~/.cache too
        backend = DynamicPyTorchUNetBackend(
            Path(self.directory.parent.parent / md_name)
        )
        backend.load()
        state_dict: Dict[str, Any] = backend.backend_object.state_dict()  # type: ignore
        state_dict = self._map_unet_state_dict(state_dict)
        self.model = init_ait_module(
            "UNet2DConditionModel", str(self.directory), state_dict
        )
        backend.unload()
        del state_dict

    def unload(self):
        del self.backend_object

    def capabilities(self) -> BaseUNetBackend.UNetCapabilities:
        return BaseUNetBackend.UNetCapabilities(
            weighted_prompts=True,
            dynamic_sizes=True,
            controlnet=False,
            width=None,
            height=None,
        )

    @property
    def backend_object(self) -> Any:
        return self.model

    @backend_object.deleter
    def backend_object(self):
        self.model = None  # type: ignore

    def _map_unet_state_dict(
        self, state_dict: Dict[str, Any], dim: int = 320
    ) -> Dict[str, Any]:
        params_ait = {}
        for key, arr in state_dict.items():
            arr = arr.to("cuda", dtype=torch.float16)
            if len(arr.shape) == 4:
                arr = arr.permute((0, 2, 3, 1)).contiguous()
            elif key.endswith("ff.net.0.proj.weight"):
                w1, w2 = arr.chunk(2, dim=0)
                params_ait[key.replace(".", "_")] = w1
                params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
                continue
            elif key.endswith("ff.net.0.proj.bias"):
                w1, w2 = arr.chunk(2, dim=0)
                params_ait[key.replace(".", "_")] = w1
                params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
                continue
            params_ait[key.replace(".", "_")] = arr

        params_ait["arange"] = (
            torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
        )
        return params_ait

    def _no_controlnet(
        self,
        latent_model_input: Any,
        timesteps: Any,
        encoder_hidden_states: Any,
        width: int,
        height: int,
    ) -> Any:
        timesteps_pt = timesteps.expand(2)  # self.batch * 2
        inputs = {
            "input0": nhwc(latent_model_input),
            "input1": timesteps_pt.cuda().half(),
            "input2": encoder_hidden_states.cuda().half(),
        }
        ys = []
        num_outputs = len(self.model.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = self.model.get_output_maximum_shape(i)
            shape[0] = 2  # self.batch * 2
            shape[1] = height // 8
            shape[2] = width // 8
            ys.append(torch.empty(shape).cuda().half())
        self.model.run_with_tensors(inputs, ys, graph_mode=False)
        noise_pred = nchw(ys[0])
        return noise_pred
