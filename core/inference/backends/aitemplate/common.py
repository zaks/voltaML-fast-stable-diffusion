import os
from typing import Any, Dict, Optional

from aitemplate.compiler import Model
import torch


def init_ait_module(
    model_name: str, workdir: str, state_dict: Optional[Dict[str, Any]]
):
    "Initiate a new compiled AITemplate model part."
    mod = Model(os.path.join(workdir, model_name, "test.so"))
    if state_dict is not None:
        mod.set_many_constants_with_tensors(state_dict)
        mod.fold_constants()
    return mod


def nhwc(tensor: torch.Tensor):
    "Basically just .permute((0, 2, 3, 1))"
    return tensor.permute((0, 2, 3, 1)).contiguous().cuda().half()


def nchw(tensor: torch.Tensor):
    "Basically just .permute((0, 3, 1, 2))"
    return tensor.permute((0, 3, 1, 2)).float()
