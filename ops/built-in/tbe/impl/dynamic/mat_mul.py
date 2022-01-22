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
dynamic mat_mul
"""
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.dynamic.batch_matmul_v2 import batch_matmul_compute
from impl.dynamic.batch_matmul_v2 import batch_matmul_v2_fuse_compute
from impl.dynamic.batch_matmul_v2 import gen_op_select_format_params
from impl.dynamic.batch_matmul_v2 import batch_matmul_v2_generalization
from impl.dynamic.batch_matmul_v2 import get_none_range_flag


# General limitation of the size for input shape: 2**32 - 1
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)
MKN_MIN = 1
MKN_MAX = 2147483648
BLOCK_CUBE = 16
DYNAMIC_FLAG = -1
DYNAMIC_FLAG_UNRANK = [-2]
NZ_LENGTH = 4
ND_LENGTH = 2
L1FUSION_INPUT_CTR = 2
EXPECT_FORMAT = ["FRACTAL_NZ", "ND"]


def get_op_support_info(input_x1, input_x2, bias, offset_w=None, output_y=None,
                        trans_a=False, trans_b=False, offset_x=0, kernel_name="matmul"):
    """
    get the matmul split, which only split the m and n, cannot cut k with bias

    """
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    dtype_b = input_x2.get("dtype")
    if format_a == 'FRACTAL_NZ':
        trans_a = not trans_a
    if format_b == 'FRACTAL_NZ':
        trans_b = not trans_b

    # cut m
    if not trans_a:
        m_split_list = [0, [0], [-1], [-1]]
        mk_split_list = [0, [1]]
    else:
        m_split_list = [0, [1], [-1], [-1]]
        mk_split_list = [0, [0]]
    # cut n
    if not trans_b:
        n_split_list = [[1, [1], [-1], [-1]]]
        nk_split_list = [1, [0]]
    else:
        n_split_list = [[1, [0], [-1], [-1]]]
        nk_split_list = [1, [1]]

    if bias:
        axis_reduce_list = None
        n_split_list.append([2, [0], [-1], [-1]])
    else:
        # cut k_dim which is reduce dim
        axis_reduce_list = [[util_select_op_base.ReduceInput(mk_split_list, nk_split_list),
                            util_select_op_base.ReduceOutput([0, 1, False])]]

    # cut m
    axis_split_matrix_a = [
        [util_select_op_base.SplitInput(m_split_list),
            util_select_op_base.SplitOutput([0, [1]])],
    ]
    # cut n
    axis_split_matrix_b = [
        [util_select_op_base.SplitInput(*n_split_list),
            util_select_op_base.SplitOutput([0, [0]])],
    ]

    axis_split_matrix = axis_split_matrix_a + axis_split_matrix_b
    min_l1space = util_common.cal_mini_l1_size_matmul(dtype_b)
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, min_l1space)

    return op_cal_info_in_json


def base_op_select_format(src_fp16_flag: bool, bias_fp32_flag: bool, impl_mode: str = "") -> tuple:
    """
    provide dynamic format to FE(Base processing)
    This funciton contains all basic format combinations

    return : dynamic format combination, static format combination
    """
    dyn_case_scenario_list = []
    full_case_scenario_list = []
    # The order from left to right is input1, input2, input3(bias), output
    base_case_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"), ("float16", "ND"),
                           ("int8", "ND"), ("float16", "FRACTAL_NZ"))]

    base_case_bias_fp32_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"), ("float", "ND"),
                                    ("int8", "ND"), ("float16", "FRACTAL_NZ"))]

    base_case_fp32_out_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"), ("float", "ND"),
                                    ("int8", "ND"), ("float", "FRACTAL_NZ"))]

    base_quant_case_scenario = [(("int8", "FRACTAL_NZ"), ("int8", "FRACTAL_Z"), ("int32", "ND"), ("int8", "ND"),
                                 ("int32", "FRACTAL_NZ"))]

    quant_case_scenario = [(("float", "NHWC"), ("float", "NHWC"), ("float", "NHWC"), ("int8", "ND"), ("float", "NHWC")),
                           (("float", "ND"), ("float", "ND"), ("float", "ND"), ("int8", "ND"), ("float", "ND")),
                           (("int32", "NHWC"), ("int32", "NHWC"), ("int32", "NHWC"), ("int8", "ND"), ("int32", "NHWC")),
                           (("int32", "ND"), ("int32", "ND"), ("int32", "ND"), ("int8", "ND"), ("int32", "ND"))]

    base_case_nzz_fp16_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_ZN_RNN"), ("float16", "ND"),
                                   ("int8", "ND"), ("float16", "FRACTAL_NZ"))]

    # ND input and output scenario
    nd_case_scenario = [
            (("float16", "ND"), ("float16", "ND"), ("float16", "ND"), ("int8", "ND"), ("float16", "ND")),
            (("float16", "ND"), ("float16", "FRACTAL_NZ"), ("float16", "ND"), ("int8", "ND"), ("float16", "ND"))
        ]
    nd_case_scenario = []
    nd_fp32out_scenario = [
            (("float16", "ND"), ("float16", "ND"), ("float", "ND"), ("int8", "ND"), ("float", "ND")),
            (("float16", "ND"), ("float16", "FRACTAL_NZ"), ("float", "ND"), ("int8", "ND"), ("float", "ND"))
        ]
    nd_fp32out_scenario = []

    dyn_case_scenario_list = base_case_scenario + nd_case_scenario + base_case_nzz_fp16_scenario
    # Construct scenario list for static
    if src_fp16_flag:
        if bias_fp32_flag and impl_mode == "keep_bias_fp32":
            full_case_scenario_list = base_case_bias_fp32_scenario + base_case_fp32_out_scenario + \
                nd_case_scenario + nd_fp32out_scenario
        else:
            full_case_scenario_list = base_case_scenario + base_case_fp32_out_scenario + \
                nd_case_scenario + nd_fp32out_scenario
    else:
        if bias_fp32_flag and impl_mode == "keep_bias_fp32":
            full_case_scenario_list = base_case_bias_fp32_scenario + base_quant_case_scenario + quant_case_scenario
        else:
            full_case_scenario_list = base_case_scenario + base_quant_case_scenario + quant_case_scenario
    return dyn_case_scenario_list, full_case_scenario_list


def op_select_format(input_x: dict, input_y: dict, bias: dict = None, offset_w: dict = None,
                     output_z: dict = None, trans_a: bool = False, trans_b: bool = False,
                     offset_x: int = 0, kernel_name: str = "matmul") -> str:
    """
    provide dynamic format to FE
    """
    # BatchMatMulV1 does not support offset_w
    src_dtype = input_x.get("dtype")
    src_fp16_flag = src_dtype == "float16"
    bias_fp32_flag = False
    if bias:
        bias_dtype = bias.get("dtype", "float16")
        bias_fp32_flag = (bias_dtype == "float32")
    scenario_combinations, _ = base_op_select_format(src_fp16_flag, bias_fp32_flag)

    param_list = gen_op_select_format_params(scenario_combinations, support_offset_w=True)
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@register_operator_compute("MatMul", op_mode="dynamic", support_fusion=False)
def mat_mul_fuse_compute(input_x1, input_x2, bias, offset_w, output_y,
                         trans_a=False, trans_b=False, offset_x=0,
                         kernel_name="matmul"):
    """
    matmul computer for fusion

    Parameters:
    input_x1: tensor
    input_x2: tensor
    bias: tensor or None
    offset_w: None
    output_y: dict
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
    batch_matmul_v2_fuse_compute(input_x1, input_x2, bias=bias, offset_w=offset_w, output_z=output_y,
                                 trans_a=trans_a, trans_b=trans_b, offset_x=offset_x,
                                 kernel_name=kernel_name)


@tbe_register.register_param_generalization("MatMul")
@tbe_register.register_param_generalization("MatMulV2")
def  matmul_generalization(input_x1, input_x2, bias, offset_w=None, output_y=None,
                           trans_a=False, trans_b=False, offset_x=0, kernel_name="matmul",
                           generalize_config=None):
    result = batch_matmul_v2_generalization(input_x1, input_x2, bias=bias, offset_w=offset_w, output_z=output_y,
                                            trans_a=trans_a, trans_b=trans_b, offset_x=offset_x,
                                            kernel_name=kernel_name, generalize_config=generalize_config)
    return result


@register_operator("MatMul")
@register_operator("MatMulV2")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT, para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL,
    para_check.REQUIRED_ATTR_BOOL,
    para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def mat_mul(input_x1, input_x2, bias, offset_w={}, output_y={},
            trans_a=False, trans_b=False, offset_x=0, kernel_name="matmul"):
    """
    caculating matrix multiplication with bias, C = A * B + bias
    only support input with NZ or ND format and fp16 in dynamic mode

    Parameters:
    input_x1: dict
        A dict object, dict with keys(shape, dtype, and range)
        the dtype must be fp16
        the format can be FRACTAL_NZ or ND
    input_x2: dict
        A dict object, dict with keys(shape, dtype and range)
        the dtype must be fp16
        the format can be FRACTAL_NZ or ND
    bias: dict
        A dict object, dict with keys(shape and dtype) or None
        the dtype must be fp16
        the format must be ND
    offset_w: None
        input offset_w tensor
    output_y: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format can be FRACTAL_NZ or ND
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
    if input_x2.get("format") == "FRACTAL_ZN_RNN":
        input_x2["format"] = "FRACTAL_Z"
    with tbe.compute():
        res = batch_matmul_compute(input_x1, input_x2, bias=bias, offset_w=offset_w, output_z=output_y,
                                   trans_a=trans_a, trans_b=trans_b, offset_x=offset_x, kernel_name=kernel_name,
                                   op_type="MatMulV2")

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get("op_res"))

    tensor_list = res.get("op_placeholder") + res.get("op_res")
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extent_in_infer_bound": False}
    }
    attr_cache_tiling = dict(res.get("op_res")[0].op.attrs.items()).get("cache_tiling", 0)
    if get_none_range_flag(input_x1, input_x2, bias) and int(attr_cache_tiling) == 1:
        config.get("build_args")["predicate_realize_bound"] = False
        config.get("build_args")["enable_branch_eliminator_else_case"] = False
    tbe.build(sch, config)
