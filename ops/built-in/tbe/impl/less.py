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
less
"""

import te.lang.cce as tbe
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util

# shape size limit for aicore is 2**31
SHAPE_SIZE_LIMIT = 2147483648


def _less_compare(data, shape, dtype, data_min):
    """
    if x is less than y, then return 1, else return 0.

    Parameters:
    ----------
    data : tuple
        two input data
    shape : list or tuple
        shape of input data
    dtype : str
        source data type, support float16,float32,int32,int8,uint8
    data_min : tvm.const
        the minimal data according to dtype

    Returns
    -------
    the compare result
    """
    data_zero = tbe.broadcast(tvm.const(0, dtype), shape, dtype)

    res_sub = tbe.vsub(data[1], data[0])
    res_min = tbe.vmin(res_sub, data_min)
    res_max = tbe.vmax(res_min, data_zero)

    if dtype == "float32":
        # max num of float32 is 2**126
        # but cce can only support 2**62, so use 62/62/2 to adaptor 126
        res_mul1 = tbe.vmuls(res_max, tvm.const(2**62, dtype=dtype))
        res_mul2 = tbe.vmuls(res_mul1, tvm.const(2**62, dtype=dtype))
        res = tbe.vmuls(res_mul2, tvm.const(2**2, dtype=dtype))
    elif dtype == "float16":
        # max num of float16 is 2**24
        # but cce can only support 2**12, so use 12/12 to adaptor 24
        res_mul1 = tbe.vmuls(res_max, tvm.const(2**12, dtype=dtype))
        res = tbe.vmuls(res_mul1, tvm.const(2**12, dtype=dtype))
    else:
        res = tbe.cast_to(res_max, "float16")

    return tbe.cast_to(res, "uint8", True)

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("less")
def less_compute(input_x, input_y, output_z, kernel_name="less"):
    """
    if x is less than y, then return 1, else return 0.

    Parameters:
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_x: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is less

    Returns
    -------
    the result
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)
    shape_x, shape_y, shape = shape_util.broadcast_shapes(shape_x, shape_y,
                                                          param_name_input1="input_x",
                                                          param_name_input2="input_y")
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    dtype = input_x.dtype
    if dtype in ("uint8", "int8"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")
        dtype = "float16"

    if dtype == "float32":
        # minimun num of float32 2**(-126)
        data_min = tbe.broadcast(tvm.const(2**(-126),
                                                   dtype=dtype), shape, dtype)
    elif dtype == "float16" and cce_product not in ("Ascend910","Ascend710"):
        # minimun num of float16 2**(-24)
        data_min = tbe.broadcast(tvm.const(2**(-24), dtype=dtype),
                                         shape, dtype)
    elif dtype == "float16" and cce_product in ("Ascend910","Ascend710"):
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")
        dtype = "float32"
        data_min = tbe.broadcast(tvm.const(2**(-126),
                                                   dtype=dtype), shape, dtype)
    elif dtype == "int32" and cce_product not in ("Ascend910","Ascend710"):
        data_min = tbe.broadcast(tvm.const(1, dtype=dtype),
                                         shape, dtype)
    else:
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")
        dtype = "float32"
        data_min = tbe.broadcast(tvm.const(2**(-126),
                                                   dtype=dtype), shape, dtype)
    input_x = tbe.broadcast(input_x, shape)
    input_y = tbe.broadcast(input_y, shape)

    return _less_compare((input_x, input_y), shape, dtype, data_min)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def less(input_x, input_y, output_z, kernel_name="less"):
    """
    do element-wise less operation between two input tensors

    Parameters:
    ----------
    input_x : dict
        shape and dtype of first input, support float16,float32,int32,
        int8,uint8
    input_y : dict
        shape and dtype of second input, support float16,float32,int32,
        int8,uint8
    output_x: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is less

    Returns
    -------
    None
    """
    shape_x = shape_util.scalar2tensor_one(input_x.get("shape"))
    shape_y = shape_util.scalar2tensor_one(input_y.get("shape"))
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                              param_name_input1="input_x", param_name_input2="input_y")

    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x,
                                                   shape_y)
    data_x = tvm.placeholder(shape_x, dtype=input_dtype, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=input_dtype, name="data_y")

    res = less_compute(data_x, data_y, output_z, kernel_name="less")
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    tbe.cce_build_code(sch, config)
