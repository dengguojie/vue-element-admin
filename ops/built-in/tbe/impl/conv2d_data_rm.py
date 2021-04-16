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
remove dirty data in M axis.
"""

from collections import deque
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.common.utils.errormgr import error_manager_cube as err_man_cube

COMPUTE_INDEX = [0]

@fusion_manager.register("conv2d_data_rm")
def conv2d_data_rm_compute(input_tensor, res_tensor=None):
    """
    Compute for removing dirty data of tensor in M axis.

    Parameters
    ----------
    input_tensor: input tensor.

    res_tensor: res tensor set by Tefusion.

    Returns
    -------
    output_tensor: output tensor after removing pad.
    """
    if type(input_tensor) != tvm.tensor.Tensor:
        err_man_cube.raise_err_one_para("E62006", "conv2d_data_rm", "The tpye of input should be tensor, "
                                        "but actually it is {}.".format(type(input_tensor)))
    if input_tensor.dtype not in ("int8", "float16"):
        err_man_cube.raise_err_one_para("E62006", "conv2d_data_rm", "The input_tensor dtype should be int8 or float16,"
                                           " but actually it is {}.".format(input_tensor.dtype))
    if len(input_tensor.shape) != 4:
        err_man_cube.raise_err_specific("conv2d_data_rm", "Wrong input_tensor shape {}, \
                                        the format should be [N C1 HW C0]!".format(input_tensor.shape))

    batch, co1, hw_mad, co0 = tuple(i.value for i in input_tensor.shape)

    tensor_queue = deque()
    tensor_queue.append(input_tensor)

    while tensor_queue:
        src_tensor = tensor_queue.popleft()
        if src_tensor.op.name == "mad1":
            output_hw = int(src_tensor.op.attrs["remove_pad_M"])
            if output_hw > hw_mad:
                err_man_cube.raise_err_specific("conv2d_data_rm",
                                                "output_hw {} is bigger than input_hw {} !!!".format(output_hw, hw_mad))
            output_shape = (batch, co1, output_hw, co0)
            output_tensor = tvm.compute(
                output_shape,
                lambda batch_idx, c1_idx, hw_idx, c0_idx:
                input_tensor[batch_idx, c1_idx, hw_idx, c0_idx],
                name="conv2d_data_rm_" + str(COMPUTE_INDEX[0]),
                tag="conv2d_data_rm")
            COMPUTE_INDEX[0] += 1
            return output_tensor

        if src_tensor.op.input_tensors:
            tensor_queue.extend(list(i for i in src_tensor.op.input_tensors))

    err_man_cube.raise_err_specific("conv2d_data_rm",
                                    "Cannot find remove_align_data_M information after traversing all input tensors!")
