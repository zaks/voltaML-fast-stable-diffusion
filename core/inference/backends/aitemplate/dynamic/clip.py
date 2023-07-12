from pathlib import Path
from typing import Any, Dict, Union

from aitemplate.compiler import Model
import torch

from ...base_backend import BaseCLIPBackend
from ..common import init_ait_module


class DynamicAITemplateCLIPBackend(BaseCLIPBackend):
    def __init__(self, directory: Union[Path, str]) -> None:
        self.model: Model
        self.directory = directory

    def load(self):
        # TODO: load in pt clip
        state_dict: Dict[str, Any] = dict(self.clip_pt.named_parameters())  # type: ignore
        state_dict = self._map_clip_state_dict(state_dict)
        self.model = init_ait_module("CLIPTextModel", str(self.directory), state_dict)
        del state_dict

    def unload(self):
        del self.backend_object

    @property
    def backend_object(self) -> Any:
        return self.model

    @backend_object.deleter
    def backend_object(self):
        self.model = None  # type: ignore

    def _map_clip_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        params_ait = {}
        for key, arr in state_dict.items():
            arr = arr.to("cuda", dtype=torch.float16)
            name = key.replace("text_model.", "")
            ait_name = name.replace(".", "_")
            if name.endswith("out_proj.weight"):
                ait_name = ait_name.replace("out_proj", "proj")
            elif name.endswith("out_proj.bias"):
                ait_name = ait_name.replace("out_proj", "proj")
            elif "q_proj" in name:
                ait_name = ait_name.replace("q_proj", "proj_q")
            elif "k_proj" in name:
                ait_name = ait_name.replace("k_proj", "proj_k")
            elif "v_proj" in name:
                ait_name = ait_name.replace("v_proj", "proj_v")
            params_ait[ait_name] = arr

        return params_ait

    def __call__(self, input_ids: Any, sequence_length: int = 77) -> Any:
        bs = input_ids.shape[0]
        position_ids = torch.arange(sequence_length).expand((bs, -1)).cuda()
        inputs = {
            "input0": input_ids,
            "input1": position_ids,
        }
        ys = []
        num_outputs = len(self.model.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = self.model.get_output_maximum_shape(i)
            shape[0] = 1  # self.batch
            ys.append(torch.empty(shape).cuda().half())
        self.model.run_with_tensors(inputs, ys, graph_mode=False)
        return ys[0].float()
