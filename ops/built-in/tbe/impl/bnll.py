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
bnll
  Oo_description :
  do element-wise bnll operation

  bnll(x, y, kernel_name = "bnll")

  Supportive_dtype_format :
   ["float16", "float32"]
   ["ND"]

"""
# pylint: disable=E0401
# pylint: disable=C0412
# pylint: disable=W0613
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check

CONST_ZERO = 0.0
CONST_ONE = 1.0
CONST_NEGATIVE_ONE = -1.0


def _bnll_compute(data, dtype):

    scalar_zero = tvm.const(CONST_ZERO, dtype)

    negative_data = tbe.vmins(data, scalar_zero)
    positive_data = tbe.vmaxs(data, scalar_zero)

    data_reverse = tbe.vaxpy(positive_data, negative_data, tvm.const(CONST_NEGATIVE_ONE, dtype))

    res = tbe.vexp(data_reverse)
    res = tbe.vadds(res, tvm.const(CONST_ONE, dtype))
    res = tbe.vlog(res)
    res = tbe.vadd(res, positive_data)

    return res


@tbe_platform.fusion_manager.fusion_manager.register("bnll")
def _bnll_computer(input_x, product):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "bnll"

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype

    if dtype == "float16" and product not in ("Hi3796CV300ES", "Hi3796CV300CS"):
        input_x = tbe.cast_to(input_x, "float32")
        d_dtype = "float32"
    else:
        d_dtype = "float16"

    res = _bnll_compute(input_x, d_dtype)

    if dtype == "float16" and product not in ("Hi3796CV300ES", "Hi3796CV300CS"):
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def bnll(input_x, output_y, kernel_name="bnll"):
    """
    calculating data
    algrithm: y=x+log(1+exp(-x)) if x>0; y=log(1+exp(x)) otherwise

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "bnll"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    input_dtype = dtype.lower()
    para_check.check_shape(shape, param_name="x")

    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="x")
    product = tbe_platform.get_soc_spec("SOC_VERSION")
    if product in ["Ascend310", "Hi3796CV300ES", "Hi3796CV300CS"] and \
        input_dtype == "float32":
        error_info = {'errCode': 'E80008', 'param_name': 'input_x', 'op_name': 'bnll', 'expect_value': "float16",
                      'real_value': input_dtype}
        raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s dtype "
                                       "should be [%s], but actually is [%s]."
                           % (error_info['op_name'], error_info['param_name'],\
                              error_info['expect_value'], error_info['real_value']))

    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape)
    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=input_dtype)

    res = _bnll_computer(data_input, product)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "bool_storage_as_1bit": False,
              "tensor_list": [data_input, res]}

    tbe.cce_build_code(schedule, config)
