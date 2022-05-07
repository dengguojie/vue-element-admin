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
dynamic batch_matmul
"""
from impl.dynamic.batch_matmul_v2 import batch_matmul_v2
from impl.dynamic.batch_matmul_v2 import check_batch_range
from impl.dynamic.batch_matmul_v2 import check_fp32_case_scenario
from impl.dynamic.batch_matmul_v2 import gen_op_select_format_params
from impl.dynamic.batch_matmul_v2 import get_op_support_info as get_op_support_info_v2
from impl.dynamic.batch_matmul_v2 import batch_matmul_v2_generalization
from impl.util import util_gemm
from impl.util import util_select_op_base
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tbe_platform


# General limitation of the size for input shape: 2**31 - 1
SHAPE_SIZE_LIMIT = 2147483647
BLOCK_CUBE = 16
DYNAMIC_FLAG = -1
DYNAMIC_FLAG_UNRANK = [-2]
BATCH_NZ_LENGTH = 5
BATCH_ND_LENGTH = 3
ND_LENGTH = 2
L1FUSION_INPUT_CTR = 2
SUPPORT_FORMAT = ["FRACTAL_NZ", "ND"]
FUZZY_SUCC_LEN = 8


def base_op_select_format(input_x, input_y, src_dtype, trans_b, src_fp16_flag: bool) -> tuple:
    """
    provide dynamic format to FE(Base processing)
    This funciton contains all basic format combinations

    return : dynamic format combination, static format combination
    """
    shape_a = input_x.get("ori_shape")
    shape_b = input_y.get("ori_shape")
    dyn_case_scenario_list = []
    full_case_scenario_list = []

    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)

    # The order from left to right is input1, input2, input3(bias), output
    base_case_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"),
                           ("float16", "ND"), ("float16", "FRACTAL_NZ"))]
    fp32_out_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"),
                          ("float", "ND"), ("float", "FRACTAL_NZ"))]

    fp32_dtype_scenario = [(("float", "NHWC"), ("float", "NHWC"), ("float", "NHWC"), ("float", "NHWC")),
                           (("float", "ND"), ("float", "ND"), ("float", "ND"), ("float", "ND"))]
    int32_dtype_scenario = [(("int32", "NHWC"), ("int32", "NHWC"), ("int32", "NHWC"), ("int32", "NHWC")),
                            (("int32", "ND"), ("int32", "ND"), ("int32", "ND"), ("int32", "ND"))]

    if not check_fp32_case_scenario(shape_a, shape_b, trans_b, src_dtype):
        fp32_dtype_scenario = []
        int32_dtype_scenario = []

    # ND input and output scenario
    nd_case_scenario = [(("float16", "ND"), ("float16", "ND"), ("float16", "ND"), ("float16", "ND")),
                        (("float16", "ND"), ("float16", "FRACTAL_NZ"), ("float16", "ND"), ("float16", "ND"))]
    nd_case_scenario = []
    nd_fp32out_scenario = [(("float16", "ND"), ("float16", "ND"), ("float", "ND"), ("float", "ND")),
                           (("float16", "ND"), ("float16", "FRACTAL_NZ"), ("float", "ND"), ("float", "ND"))]
    nd_fp32out_scenario = []
    cube_vector_scenario = [
        (("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"), ("float32", "ND"), ("float16", "FRACTAL_NZ")),
        (("float32", "FRACTAL_NZ"), ("float32", "FRACTAL_NZ"), ("float32", "ND"), ("float32", "FRACTAL_NZ")),
        (("int8", "FRACTAL_NZ"), ("int8", "FRACTAL_NZ"), ("int32", "ND"), ("int32", "FRACTAL_NZ")),
        (("int8", "FRACTAL_NZ"), ("int8", "FRACTAL_Z"), ("int32", "ND"), ("int32", "FRACTAL_NZ")),
        (("bfloat16", "FRACTAL_NZ"), ("bfloat16", "FRACTAL_NZ"), ("float32", "ND"), ("bfloat16", "FRACTAL_NZ"))
    ]
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    support_s322f32 = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "s322f32")

    dyn_case_scenario_list = base_case_scenario + nd_case_scenario
    if dynamic_flag and not check_batch_range(input_x, input_y):
        warnings.warn("input_x, input_y out of batch_range")
        dyn_case_scenario_list = []
    # Construct scenario list for static
    if support_l0c2out:
        full_case_scenario_list = cube_vector_scenario
    elif src_fp16_flag:
        full_case_scenario_list = base_case_scenario + fp32_out_scenario + nd_case_scenario + nd_fp32out_scenario
    else:
        full_case_scenario_list = base_case_scenario + fp32_dtype_scenario
        if support_s322f32:
            full_case_scenario_list += int32_dtype_scenario
    return dyn_case_scenario_list, full_case_scenario_list


def op_select_format(input_x: dict, input_y: dict, bias: dict = None, output_z: dict = None, trans_a: bool = False,
                     trans_b: bool = False, kernel_name: str = "matmul") -> str:
    """
    provide dynamic format to FE
    """
    src_dtype = input_x.get("dtype")
    src_fp16_flag = src_dtype == "float16"
    scenario_combinations, _ = base_op_select_format(input_x, input_y, src_dtype, trans_b, src_fp16_flag)

    param_list = gen_op_select_format_params(scenario_combinations, support_offset_w=False)
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def get_op_support_info(input_x1, input_x2, bias=None, output_z=None,
                        trans_a=False, trans_b=False, kernel_name="matmul"):
    """
    get the batch_matmul split, which only split batch, m and n, cannot cut k with bias
    """
    op_cal_info_in_json = get_op_support_info_v2(
        input_x1, input_x2, bias=bias, output_z=output_z, trans_a=trans_a, trans_b=trans_b, kernel_name=kernel_name)
    return op_cal_info_in_json


@register_operator_compute("BatchMatMul", op_mode="dynamic", support_fusion=False)
def batch_matmul_fuse_compute(input_x1, input_x2, bias, output_z,
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
        the format must be FRACTAL_NZ or ND
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



@tbe_register.register_param_generalization("BatchMatMul")
def batch_matmul_generalization(input_x1, input_x2, bias=None, output_z=None,
                                trans_a=False, trans_b=False, kernel_name="batchmatmul",
                                generalize_config=None):
    result = None
    if generalize_config.get("mode") == "keep_rank":
        result = batch_matmul_v2_generalization(input_x1, input_x2, bias=bias, output_z=output_z,
                                                trans_a=trans_a, trans_b=trans_b, kernel_name=kernel_name,
                                                generalize_config=generalize_config)
        # If pass fuzzy compile check, delete redundancy info, e.g offset_w, offset_x
        if isinstance(result, list) and len(result[0]) == FUZZY_SUCC_LEN:
            input_x1, input_x2, bias, _, output_z, trans_a, trans_b, _ = result[0]
            single_op_flag = True if generalize_config.get("single_op") == 'true' else False
            set_range_none = single_op_flag and not bias
            if set_range_none:
                ori_range_1 = util_gemm.cal_gemm_shape_range(input_x1["ori_shape"],
                    input_x1["ori_format"], set_range_none)
                ori_range_2 = util_gemm.cal_gemm_shape_range(input_x2["ori_shape"],
                    input_x2["ori_format"], set_range_none)
                input_x1["ori_range"], input_x2["ori_range"] = ori_range_1, ori_range_2
            result = [[input_x1, input_x2, bias, output_z, trans_a, trans_b]]
    # binary generalization mode
    elif generalize_config.get("mode") == "all_shape":
        result = []
        shape_x1 = util_gemm.cal_gemm_shape_binary(input_x1)
        shape_x2 = util_gemm.cal_gemm_shape_binary(input_x2)
        shape_z = util_gemm.cal_gemm_shape_binary(output_z)
        input_x1["shape"] = shape_x1
        input_x2["shape"] = shape_x2
        output_z["shape"] = shape_z
        result.append([input_x1, input_x2, bias, output_z, trans_a, trans_b])
    return result


@register_operator("BatchMatMul")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_BOOL, para_check.REQUIRED_ATTR_BOOL,
    para_check.KERNEL_NAME)
def batch_matmul(input_x1, input_x2, bias=None, output_z=None,
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
    batch_matmul_v2(input_x1, input_x2, bias=bias, output_z=output_z,
                    trans_a=trans_a, trans_b=trans_b, kernel_name=kernel_name)
