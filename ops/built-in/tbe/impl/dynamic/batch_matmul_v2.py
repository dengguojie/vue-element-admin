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
from impl.util import util_gemm
from impl.dynamic.batch_matmul import batch_matmul
from impl.dynamic.batch_matmul import batch_matmul_fuse_compute
from impl.dynamic.batch_matmul import get_op_support_info as get_op_support_info_batchmatmul
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_register


def get_op_support_info(input_x1, input_x2, bias=None, offset_w=None, output_z=None,
                        trans_a=False, trans_b=False, offset_x=0, kernel_name="matmul"):
    """
    get the batch_matmul_v2 split, which only split batch, m and n, cannot cut k with bias

    """
    op_cal_info_in_json = get_op_support_info_batchmatmul(
        input_x1, input_x2, bias, output_z, trans_a, trans_b, kernel_name
    )
    return op_cal_info_in_json


@register_operator_compute("BatchMatMul", op_mode="dynamic", support_fusion=False)
def batch_matmul_v2_fuse_compute(input_x1, input_x2, bias=None, offset_w=None, output_z={},
                                 trans_a=False, trans_b=False, offset_x=0,
                                 kernel_name="matmul"):
    """
    matmul computer for fusion

    Parameters:
    input_x1: tensor
    input_x2: tensor
    bias: tensor or None
    offset_w: tensor or None
    output_z: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    trans_a: bool
        If true, shape_a == transposed before multiplication
    trans_b: bool
        If true, shape_a == transposed before multiplication
    offset_x: int
        offset of gradients in quant mode
    kernel_name: str
        cce kernel_name
    Returns
    -------
    res : dict
        A dict object, dict with input tensor and output tensor
    """
    return batch_matmul_fuse_compute(input_x1, input_x2, bias, output_z, trans_a, trans_b, kernel_name)


@tbe_register.register_param_generalization("BatchMatMulV2")
def batch_matmul_v2_generalization(input_x1, input_x2, bias=None, offset_w=None, output_z=None,
                                   trans_a=False, trans_b=False, offset_x=0, kernel_name="batch_matmul",
                                   generalize_config={"mode": "keep_rank"}):
    result = []
    if generalize_config["mode"] == "keep_rank": # fuzzy compile
        # get range generalization
        ori_range_x1 = util_gemm.cal_gemm_shape_range(input_x1["ori_shape"], input_x1["ori_format"])
        ori_range_x2 = util_gemm.cal_gemm_shape_range(input_x2["ori_shape"], input_x2["ori_format"])
        util_gemm.generalize_input_keep_rank_gemm(input_x1)
        util_gemm.generalize_input_keep_rank_gemm(input_x2)
        input_x1["ori_range"], input_x2["ori_range"] = ori_range_x1, ori_range_x2
        if bias:
            ori_range_bias = util_gemm.cal_gemm_shape_range(bias["ori_shape"], bias["ori_format"])
            util_gemm.generalize_input_keep_rank_gemm(bias)
            bias["ori_range"] = ori_range_bias
        util_gemm.generalize_input_keep_rank_gemm(output_z)
        result.append([input_x1, input_x2, bias, offset_w, output_z, {"trans_a": trans_a}, {"trans_b": trans_b},
                       {"offset_x": offset_x}])
    else:
        error_manager_cube.raise_err_one_para(
            "E62306",
            "BatchMatMulV2",
            "Invalid generalize mode, currently only support keep_rank"
        )

    return result


@register_operator("BatchMatMulV2")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
    para_check.OPTION_INPUT,para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL,
    para_check.REQUIRED_ATTR_BOOL, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def batch_matmul_v2(input_x1, input_x2, bias=None, offset_w={}, output_z={},
                    trans_a=False, trans_b=False, offset_x=0, kernel_name="matmul"):
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
    offset_w: dict
        A dict object, dict with keys(shape and dtype) or None
    output_z: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    trans_a: bool
        If true, shape_a == transposed before multiplication
    trans_b: bool
        If true, shape_a == transposed before multiplication
    offset_x: int
        offset of gradients in quant mode
    kernel_name: str
        cce kernel_name
    Returns
    -------
    res : dict
        None
    """
    batch_matmul(input_x1, input_x2, bias, output_z, trans_a, trans_b, kernel_name)
