#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import torch
from aitemplate.compiler import compile_model
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

from ..modeling.controlnet_unet_2d_condition import (
    UNet2DConditionModel as ait_UNet2DConditionModel,
)
from .util import mark_output


def map_unet_params(pt_mod, dim):
    pt_params = dict(pt_mod.named_parameters())
    params_ait = {}
    for key, arr in pt_params.items():
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


def compile_controlnet_unet(  # pylint: disable=dangerous-default-value
    pt_mod,
    dump_dir: str = "./tmp",
    batch_size: int = 2,
    height: int = 64,
    width: int = 64,
    dim: int = 320,
    hidden_dim: int = 1024,
    use_fp16_acc: bool = False,
    convert_conv_to_gemm: bool = False,
    attention_head_dim=[5, 10, 20, 20],  # noqa: B006
):
    "Compile UNet to AIT"

    ait_mod = ait_UNet2DConditionModel(
        sample_size=64,
        cross_attention_dim=hidden_dim,
        attention_head_dim=attention_head_dim,
    )
    ait_mod.name_parameter_tensor()

    # set AIT parameters
    pt_mod = pt_mod.eval()
    params_ait = map_unet_params(pt_mod, dim)

    latent_model_input_ait = Tensor(
        [batch_size, height, width, 4], name="input0", is_input=True  # type: ignore
    )
    timesteps_ait = Tensor([batch_size], name="input1", is_input=True)  # type: ignore
    text_embeddings_pt_ait = Tensor(
        [batch_size, 64, hidden_dim], name="input2", is_input=True  # type: ignore
    )

    width_d_8 = width // 8
    height_d_8 = height // 8
    width_d_16 = width // 16
    height_d_16 = height // 16
    width_d_32 = width // 32
    height_d_32 = height // 32
    width_d_64 = width // 64
    height_d_64 = height // 64

    dbar_0_pt_ait = Tensor([batch_size, height_d_8, width_d_8, 320], name="input3", is_input=True)  # type: ignore
    dbar_1_pt_ait = Tensor([batch_size, height_d_8, width_d_8, 320], name="input4", is_input=True)  # type: ignore
    dbar_2_pt_ait = Tensor([batch_size, height_d_8, width_d_8, 320], name="input5", is_input=True)  # type: ignore
    dbar_3_pt_ait = Tensor([batch_size, height_d_16, width_d_16, 320], name="input6", is_input=True)  # type: ignore
    dbar_4_pt_ait = Tensor([batch_size, height_d_16, width_d_16, 640], name="input7", is_input=True)  # type: ignore
    dbar_5_pt_ait = Tensor([batch_size, height_d_16, width_d_16, 640], name="input8", is_input=True)  # type: ignore
    dbar_6_pt_ait = Tensor([batch_size, height_d_32, width_d_32, 640], name="input9", is_input=True)  # type: ignore
    dbar_7_pt_ait = Tensor([batch_size, height_d_32, width_d_32, 1280], name="input10", is_input=True)  # type: ignore
    dbar_8_pt_ait = Tensor([batch_size, height_d_32, width_d_32, 1280], name="input11", is_input=True)  # type: ignore
    dbar_9_pt_ait = Tensor([batch_size, height_d_64, width_d_64, 1280], name="input12", is_input=True)  # type: ignore
    dbar_10_pt_ait = Tensor([batch_size, height_d_64, width_d_64, 1280], name="input13", is_input=True)  # type: ignore
    dbar_11_pt_ait = Tensor([batch_size, height_d_64, width_d_64, 1280], name="input14", is_input=True)  # type: ignore
    mid_block_additional_residual_pt_ait = Tensor(
        [batch_size, height_d_64, width_d_64, 1280],  # type: ignore
        name="input15",
        is_input=True,
    )

    Y = ait_mod(
        latent_model_input_ait,
        timesteps_ait,
        text_embeddings_pt_ait,
        dbar_0_pt_ait,
        dbar_1_pt_ait,
        dbar_2_pt_ait,
        dbar_3_pt_ait,
        dbar_4_pt_ait,
        dbar_5_pt_ait,
        dbar_6_pt_ait,
        dbar_7_pt_ait,
        dbar_8_pt_ait,
        dbar_9_pt_ait,
        dbar_10_pt_ait,
        dbar_11_pt_ait,
        mid_block_additional_residual_pt_ait,
    )
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y,
        target,
        dump_dir,
        "ControlNetUNet2DConditionModel",
        constants=params_ait,
    )
