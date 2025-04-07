# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# https://huggingface.co/runwayml/stable-diffusion-v1-5
from diffusers import FluxPipeline
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import tracemalloc
import torch


class ThisTester(ModelTester):
    def _load_model(self):
        model_id = "black-forest-labs/FLUX.1-schnell"
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        return pipe

    def _load_inputs(self):
        prompt = [
            "A cat holding a sign that says hello world",
        ]
        arguments = {
            "prompt": prompt,
            "guidance_scale": 0.0,
            "height": 768,
            "width": 1360,
            "num_inference_steps": 4,
            "max_sequence_length": 256,
        }
        return arguments


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_flux(record_property, mode, op_by_op):
    model_name = "Flux"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO
    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group="red",
    )
    results = tester.test_model()
    if mode == "eval":
        image = results.images[0]

    tester.finalize()
