# Copyright 2021 Huawei Technologies Co., Ltd
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
dynamic depth_to_space
"""
from impl.dynamic.transpose import Transpose
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info

CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
BLOCK_SIZE = 32
MAX_INT64_VALUE = 2**64 - 1
TILING_MAX_SIZE_GM = 2048  # 16KB


def get_op_support_info(x, y, block_size, data_format='NHWC', kernel_name="depth_to_space"):
    """
    get_op_support_info
    """
    format_x = x.get("format").upper()
    if format_x == "NHWC":
        axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])],
                             [SplitInput([0, [1], [-1], [-1]]), SplitOutput([0, [1]])],
                             [SplitInput([0, [2], [-1], [-1]]), SplitOutput([0, [2]])]]
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# pylint: disable=invalid-name,unused-argument,too-many-locals,protected-access
@register_operator("DepthToSpace")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def depth_to_space(x, y, block_size, data_format='NHWC', kernel_name="depth_to_space"):
    """
    the main function of depth_to_space

    Parameters
    ----------
    x: dict
        dict with keys(shape, dtype) of input
    y: dict
        dict with keys(shape, dtype) of output
    block_size: int
        the size of the spatial block
    data_format: str
        data format, default value is "NHWC"
    kernel_name: str
        kernel name, default value is "depth_to_space"

    Returns
    -------
    tik_instance: tik_instance
    """
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    para_check.check_shape(input_shape, param_name="x")
    check_list = ("int8", "int16", "int32", "uint8", "uint16", "uint32", "uint64", "int64", "float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="x")

    # run tick
    tik_inst = tik.Tik()
    data_in = tik_inst.Tensor(input_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "data_in")
    data_out = tik_inst.Tensor(input_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "data_out")
    data_workspace = tik_inst.Tensor(input_dtype, (1024,), tik.scope_gm, "data_workspace", is_workspace=True)
    data_tiling = tik_inst.Tensor("int64", (TILING_MAX_SIZE_GM,), tik.scope_gm, "data_tiling")
    tensor_list = [data_in, None, data_out, data_workspace, data_tiling]
    obj = Transpose(tik_inst, input_dtype, tensor_list, kernel_name)
    obj.compute_tiling()

    tbe_context.get_context().add_compile_info("vars", {
        "ub_size": UB_SIZE // BLOCK_SIZE,
        "core_num": CORE_NUM,
        "dtype": input_dtype,
        "block_size": block_size,
    })
    # this "global_variable_link" flag suggest ccec.py do link without "-r" option
    # which will result in global variable in cce file with wrong address
    tbe_context.get_context().add_compile_info("global_variable_link", True)
    opt_config = {"enable_const_fold": True}
    obj.tik_inst.BuildCCE(kernel_name=obj.kernel_name,
                          inputs=[obj.data_in],
                          outputs=[obj.data_out],
                          flowtable=[obj.data_tiling],
                          config=opt_config)

    return obj.tik_inst
