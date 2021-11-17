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
split
"""
from __future__ import absolute_import
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from .split_v import SplitV


def check_input_params(x, split_dim, y, num_split):
    """
    check input parameters
    """
    # split has 2 input tensors, so 62 is the maximum of output tensors
    if num_split > 62 or num_split < 1:
        error_manager_vector.raise_err_input_value_invalid("split", "num_split",
                                                           "62 is the maximum of num_split", num_split)

    x_dtype = x.get("dtype").lower()
    split_dim_dtype = split_dim.get("dtype").lower()
    output_dtype = y[0].get("dtype").lower()

    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32")
    para_check.check_dtype(x_dtype, check_list, param_name="x")
    para_check.check_dtype(split_dim_dtype, ("int32",), param_name="split_dim")

    if x_dtype != output_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("split", "x_dtype", "y_dtype",
                                                              x_dtype, output_dtype)


@register_operator("Split")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.DYNAMIC_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def split(split_dim, x, y, num_split, kernel_name="split"):
    """
    Split a tensor into num_split tensors along one dimension.

    Parameters
    ----------
    split_dim: dict
        the dict of input split_dim tensor.
        An int, specifies the dimension along which to split.
    x: dict
        the dict of input tensor.
    y: list or tuple
        the list of output tensor.
    num_split: int
        an integer indicating the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split".

    Returns
    -------
    compile info
    """
    check_input_params(x, split_dim, y, num_split)

    size_splits = {}
    obj = SplitV(x, size_splits, split_dim, y, num_split, kernel_name)
    obj.split_v_compute_tiling()

    tik_inst = obj.tik_instance
    tik_inst.BuildCCE(kernel_name=obj.kernel_name,
                      inputs=(obj.split_dim_gm, obj.x_gm),
                      outputs=obj.outputs_gm,
                      flowtable=(obj.tiling_gm,), enable_l2=True)

    # add compile info
    tbe_context.get_context().add_compile_info("vars", {"core_num": obj.core_num,
                                    "ub_elems": obj.ub_elems,
                                    "num_split": obj.num_split
                                    })

    return tik_inst
