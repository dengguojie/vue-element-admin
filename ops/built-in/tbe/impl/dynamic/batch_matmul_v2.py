# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
dynamic batch_matmul_v2
"""
import te.lang.base as tbe_base
from te.utils import para_check
from impl.dynamic.batch_matmul import batch_matmul
from impl.dynamic.batch_matmul import batch_matmul_fuse_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute

@register_operator_compute("BatchMatMul", op_mode="dynamic", support_fusion=False)
def batch_matmul_v2_fuse_compute(input_x1, input_x2, bias, output_z,
                         trans_a=False, trans_b=False,
                         kernel_name="matmul"):
    """
    matmul computer for fusion

    Parameters:
    input_x1: tensor
    input_x2: tensor
    bias: tensor or None
    output_z: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    trans_a: bool
        If true, shape_a == transposed before multiplication
    trans_b: bool
        If true, shape_a == transposed before multiplication
    kernel_name: str
        cce kernel_name
    Returns
    -------
    res : dict
        A dict object, dict with input tensor and output tensor
    """
    return batch_matmul_fuse_compute(input_x1, input_x2, bias, output_z, trans_a, trans_b, kernel_name)


@register_operator("BatchMatMulV2")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_BOOL, para_check.REQUIRED_ATTR_BOOL,
    para_check.KERNEL_NAME)
def batch_matmul_v2(input_x1, input_x2, bias=None, output_z={},
                 trans_a=False, trans_b=False, kernel_name="matmul"):
    """
    caculating matrix multiplication with bias, C = A * B + bias
    only support input with nz format and fp16 in dynamic mode

    Parameters:
    input_x1: dict
        A dict object, dict with keys(shape, dtype, and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    input_x2: dict
        A dict object, dict with keys(shape, dtype and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    bias: dict
        A dict object, dict with keys(shape and dtype) or None
        the dtype must be fp16
        the format must be ND
    output_z: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    trans_a: bool
        If true, shape_a == transposed before multiplication
    trans_b: bool
        If true, shape_a == transposed before multiplication
    kernel_name: str
        cce kernel_name
    Returns
    -------
    res : dict
        None
    """
    batch_matmul(input_x1, input_x2, bias, output_z, trans_a, trans_b, kernel_name)
