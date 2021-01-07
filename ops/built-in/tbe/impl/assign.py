# Copyright 2019 Huawei Technologies Co., Ltd
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
assign
"""
import operator

import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector


# pylint: disable=unused-argument,invalid-name,consider-using-enumerate
def _check_params(ref_shape, value_shape, dtype, kernel_name):
    """
    check the parameters including ref_shape, value_shape, dtype and kernel_name

    Parameters
    ----------
    ref_shape: list or tuple
        shape of ref_tensor
    value_shape: list or tuple
        shape of value_tensor
    dtype: str
        the data type
    kernel_name: str
        cce kernel name, default value is "cce_assign"

    Returns
    -------
    None
    """
    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="ref")

    if operator.ne(list(ref_shape), list(value_shape)):
        error_detail = "Shape of ref and value should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("assign", "ref", "value", error_detail)

    para_check.check_shape(ref_shape, param_name="ref")


def _assign_schedule(res, tensor_val):
    """
    assign schedule

    Parameters
    ----------
    res: result of compute
    tensor_val: tensor val

    Returns
    -------
    output sch
    """
    def _ceil(m, n):
        return (m + n - 1) // n

    def _tiling(shape, dtype):
        ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        dtype_bytes_size = tbe_platform.get_bit_len(dtype) // 8
        # only use 1/2 ub
        total_ele = ub_size_bytes // dtype_bytes_size // 2
        core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        # 1 block is 32B
        block_ele = 32 // dtype_bytes_size

        fused_axis_factor = shape[0]
        one_core_limit = dtype_bytes_size * shape[0]
        # 35000 is an experience number, the performance is better under one core.
        if fused_axis_factor >= core_num and one_core_limit > 35000:
            fused_axis_factor = _ceil(fused_axis_factor, core_num)
            fused_axis_factor = _ceil(fused_axis_factor, block_ele) * block_ele
        total_ele = ((total_ele + block_ele) // block_ele) * block_ele
        fused_factor = min(fused_axis_factor, total_ele)
        return fused_axis_factor, fused_factor
    # set ub
    tensor_input = tensor_val
    x_shape = [i.value for i in tensor_input.shape]
    core_factor, ub_factor = _tiling(x_shape, tensor_input.dtype)
    sch = tvm.create_schedule(res.op)
    tensor_input_in_ub = sch.cache_read(tensor_input, tbe_platform.scope_ubuf, [res])

    # set axis info
    axis_core_out, axis_core_in = sch[res].split(res.op.axis[0], core_factor)
    axis_ub_out, axis_ub_in = sch[res].split(axis_core_in, ub_factor)
    sch[tensor_input_in_ub].compute_at(sch[res], axis_ub_out)

    # set ping pong
    sch[tensor_input_in_ub].double_buffer()

    # set multi cores
    block = tvm.thread_axis("blockIdx.x")
    sch[res].bind(axis_core_out, block)

    # set emit_insn
    sch[tensor_input_in_ub].emit_insn(tensor_input_in_ub.op.axis[0], tbe_platform.DMA_COPY)
    sch[res].emit_insn(axis_ub_in, tbe_platform.DMA_COPY)
    return sch


# pylint: disable=locally-disabled,too-many-arguments,unnecessary-lambda
# pylint: disable=locally-disabled,too-many-branches,too-many-locals
# pylint: disable=locally-disabled,unused-argument,too-many-statements
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def assign(ref, value, output, kernel_name="assign"):
    """
    algorithm: assign
    calculating: update 'ref' by assigning 'value' to it

    Parameters
    ----------
    ref: dict
        dict of input_ref, include shape and dtype,
    value: dict
        dict of input_value, include shape and dtype,
        Must have the same shape and dtype as input_ref
    output: dict
        dict of output
    kernel_name : str
        cce kernel name, default value is assign

    Returns
    -------
    None
    """
    ref_shape = shape_util.scalar2tensor_one(ref.get("shape"))
    value_shape = shape_util.scalar2tensor_one(value.get("shape"))
    dtype = ref.get("dtype").lower()
    _check_params(ref_shape, value_shape, dtype, kernel_name)

    res_num = 1
    for i in range(len(ref_shape)):
        res_num = res_num * ref_shape[i]
    reshape = [res_num,]

    tensor_val = tvm.placeholder(reshape, dtype=dtype, name='tensor_val')
    res = tvm.compute(reshape, lambda *i: tensor_val(*i), name='res')
    sch = _assign_schedule(res, tensor_val)

    with tbe_platform.cce_build.build_config:
        tvm.build(sch, [res, tensor_val], "cce", name=kernel_name)
