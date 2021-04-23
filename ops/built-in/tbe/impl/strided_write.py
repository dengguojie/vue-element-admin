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
strided write
"""
from te import tvm
from topi import generic
from topi.cce import util
from te.platform.fusion_manager import fusion_manager
from impl.util.platform_adapter import error_manager_vector


STRIDED_WRITE_TAG = "strided_write"


# pylint: disable=invalid-name,unnecessary-lambda,unused-argument,unused-variable
def check_params(x, y, axis):
    """
    check the parameters including x, y, axis.
    """
    if len(x.get("shape")) != 5:
        error_manager_vector.raise_err_specific_reson("strided_read", "x's length must be 5 \
                                                      while length is{}.".format(len(x.get("shape"))))
    if len(y.get("shape")) != 5:
        error_manager_vector.raise_err_specific_reson("strided_read", "y's length must be 5 \
                                                      while length is{}.".format(len(y.get("shape"))))
    if x.get("dtype") not in ("float16", "int8"):
        error_manager_vector.raise_err_input_dtype_not_supported("strided_read", "x", "float16 or int8",
                                                                 x.get("dtype"))
    if y.get("dtype") not in ("float16", "int8"):
        error_manager_vector.raise_err_input_dtype_not_supported("strided_read", "y", "float16 or int8",
                                                                 y.get("dtype"))
    if x.get("format") != "NC1HWC0":
        error_manager_vector.raise_err_input_format_invalid("strided_read", "x", "NC1HWC0", x.get("format"))
    if y.get("format") != "NC1HWC0":
        error_manager_vector.raise_err_input_format_invalid("strided_read", "y", "NC1HWC0", y.get("format"))
    if x.get("dtype") != y.get("dtype"):
        error_manager_vector.raise_err_inputs_dtype_not_equal("strided_read", "x", "y",
                                                              x.get("dtype"), y.get("dtype"))
    if axis != 1:
        error_manager_vector.raise_err_input_value_invalid("strided_read", "axis", "1", str(axis))


@fusion_manager.register("strided_write")
def strided_write_compute(x, y, axis, stride, kernel_name='strided_write'):
    """
    write data to tensor by stride.

    Parameters:
    ----------
    x: placeholder of input tesnor.

    y: dict of output tensor.

    axis: which axis to write data by stride.

    stride: data write stride.

    kernel_name: cce kernel name, default value is "strided_write".

    Returns:
    ----------
    output_y: result tensor.
    """
    shape_y = tuple(i.value for i in x.shape)
    output_y = tvm.compute(shape_y, lambda *indice: x(*indice),
                           name="strided_write",
                           attrs={"stride": stride},
                           tag=STRIDED_WRITE_TAG)
    return output_y


@util.check_input_type(dict, dict, int, int, str)
def strided_write(x, y, axis, stride, kernel_name='strided_write'):
    """
    write data to tensor by stride.

    Parameters:
    ----------
    x: dict of input.

    y: dict of output.

    axis: which axis to write data by stride.

    stride: data write stride.

    kernel_name: cce kernel name, default value is "strided_write".

    Returns:
    -------
    None
    """

    check_params(x, y, axis)
    dtype_x = x.get("dtype")
    n_i, c1_i, h_i, w_i, c0_i = x.get("shape")
    shape_x = n_i, c1_i, h_i*w_i, c0_i
    input_x = tvm.placeholder(shape_x, name="input_x", dtype=dtype_x)
    res = strided_write_compute(input_x, y, axis, stride, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)
