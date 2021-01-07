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
bias_add
"""
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te import tvm
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_format
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_elewise_shape_range
from te.utils.op_utils import REQUIRED_ATTR_STR
from te.utils import shape_util


# pylint: disable=too-many-locals,unused-argument
# pylint: disable=too-many-statements,too-many-branches,invalid-name
def check_equal(a, b):
    """
    check whether a equal to b or not

    Parameters
    ----------
    a : int
    b : int
    Returns
    -------
    res : true or false
    """
    if a != -1 and b != -1 and a != b:
        return False
    return True


def bias_add_compute(x, bias, y, data_format, kernel_name="bias_add"):
    """
    calculating data's bias add

    Parameters
    ----------
    x : tvm tensor
              x data x
    bias : tvm tensor
              x data y
    y : tvm tensor
              y data
    data_format: A string.
                'N...C' and 'NC...' are supported.
    kernel_name : string
                  cce kernel name, default value is "bias_add"

    Returns
    -------
    res : y of the data's bias add
    """
    _, _, shape_max = shape_util.broadcast_shapes(shape_util.shape_to_list(x.shape),
                                                  shape_util.shape_to_list(bias.shape),
                                                  param_name_input1="x",
                                                  param_name_input2="bias")

    data_x = tbe.broadcast(x, shape_max)
    data_bias = tbe.broadcast(bias, shape_max)
    res = tbe.vadd(data_x, data_bias)

    return res


@tbe_base.register_operator("BiasAdd")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 REQUIRED_ATTR_STR, KERNEL_NAME)
def bias_add(x, bias, y, data_format="NHWC", kernel_name="bias_add"):
    """
    algorithm: bias_and
    Reduce a tensor on a certain axis based on min

    Parameters
    ----------
    x : dict
              the shape and dtype of the tensor x
    bias : dict
              the shape and dtype of the tensor y
    y :  dict
              the shape and dtype of the tensor z
    data_format: A string.
                'N...C' and 'NC...' are supported.
    kernel_name : string
                  cce kernel name, default value is "bias_add"
    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    shape_bias = bias.get("shape")
    range_bias = bias.get("range")
    range_x = x.get("range")

    dtype_x = x.get("dtype").lower()
    dtype_bias = bias.get("dtype").lower()
    dtype_y = y.get("dtype").lower()
    data_format = data_format.upper()

    check_tuple = ("float16", "float32", "int32")
    data_format_tuple = ("NCHW", "NHWC", "NDHWC", "NCDHW")
    check_dtype(dtype_x, check_tuple, param_name="x")
    check_dtype(dtype_bias, check_tuple, param_name="bias")
    check_dtype(dtype_y, check_tuple, param_name="y")
    check_format(data_format, data_format_tuple, param_name='input_format')

    if dtype_x != dtype_bias:
        raise RuntimeError("The dtype of x and bias must be the same")

    if x.get("format") is not None and x.get("format").upper() == "NC1HWC0":
        ori_format_x = x.get("ori_format").upper()
        ori_shape_x = x.get("ori_shape")
        if len(shape_x) != 5:
            raise RuntimeError("bias_add only support shape 5D, when input format is NC1HWC0")

        if ori_format_x != data_format:
            raise RuntimeError("the input ori_format and data_format must be the same")
        if bias.get("format") is not None and bias.get("format").upper() == "NC1HWC0":
            ori_shape_bias = bias.get("ori_shape")
            if ori_format_x == "NCHW" and not check_equal(ori_shape_x[1], ori_shape_bias[0]):
                raise RuntimeError("data_format is NCHW, shape_bias must "
                                   "be equal to the second axis of shape_x")
            if ori_format_x == "NHWC" and not check_equal(ori_shape_x[-1], ori_shape_bias[0]):
                raise RuntimeError("data_format is NHWC, shape_bias must "
                                   "be equal to the last axis of shape_x")
        else:
            if ori_format_x == "NCHW" and not check_equal(ori_shape_x[1], shape_bias[0]):
                raise RuntimeError("data_format is NCHW, shape_bias must "
                                   "be equal to the second axis of shape_x")
            if ori_format_x == "NHWC" and not check_equal(ori_shape_x[-1], shape_bias[0]):
                raise RuntimeError("data_format is NHWC, shape_bias must "
                                   "be equal to the last axis of shape_x")
        shape_bias = (1, shape_x[1], 1, 1, shape_x[4])
        range_bias = ((1, 1), range_x[1], (1, 1), (1, 1), range_x[4])

    elif x.get("format") is not None and x.get("format").upper() == "NDHWC":
        if len(shape_x) != 5:
            raise RuntimeError("bias_add only support shape 5D, when input format is NDHWC")

        if not check_equal(shape_x[4], shape_bias[0]):
            raise RuntimeError("data_format is NDHWC, shape_bias must"
                               "be equal to the fifth axis of shape_x")
        shape_bias = (1, ) * (len(shape_x) - 1) + (shape_x[-1], )
        range_bias = ((1, 1), ) * (len(shape_x) - 1) + (range_x[-1], )

    elif x.get("format") is not None and x.get("format").upper() == "NCDHW":
        if len(shape_x) != 5:
            raise RuntimeError("bias_add only support shape 5D, "
                               "when input format is NCDHW")
        if not check_equal(shape_x[1], shape_bias[0]):
            raise RuntimeError("data_format is NCDHW, shape_bias must "
                               "be equal to the second axis of shape_x")
        shape_bias = (1, shape_x[1]) + (1, ) * (len(shape_x) - 2)
        range_bias = ((1, 1), range_x[1]) + ((1, 1), ) * (len(shape_x) - 2)

    elif x.get("format") is not None and x.get("format").upper() == "NDC1HWC0":
        if len(shape_x) != 6:
            raise RuntimeError("bias_add only support shape 6D"
                               "when input format is NDC1HWC0")
        ori_shape_x = x.get("ori_shape")
        if x.get("ori_format").upper() == "NDHWC":
            if not check_equal(ori_shape_x[4], shape_bias[0]):
                raise RuntimeError("data_format is NDHWC, shape_bias must "
                                   "be equal to the fifth axis of shape_x")
        elif x.get("ori_format").upper() == "NCDHW":
            if not check_equal(ori_shape_x[1], shape_bias[0]):
                raise RuntimeError("data_format is NCDHW, shape_bias must "
                                   "be equal to the second axis of shape_x")
        shape_bias = (1, 1, shape_x[2], 1, 1, shape_x[5])
        range_bias = ((1, 1), (1, 1), range_x[2], (1, 1), (1, 1), range_x[5])

    else:
        if data_format == "NCHW":
            if len(shape_x) < 2 or len(shape_x) > 4:
                raise RuntimeError("bias_add only support shape 2D to 4D when input format is NCHW")
            if not check_equal(shape_x[1], shape_bias[0]):
                raise RuntimeError("data_format is NCHW, shape_bias must"
                                   " be equal to the second axis of shape_x"
                                   ", but {} and {}".format(shape_bias[0], shape_x[1]))
            shape_bias = (1, shape_x[1],)
            range_bias = ((1, 1), range_x[1],)
            for i in range(2, len(shape_x)):
                shape_bias = shape_bias + (1,)
                range_bias = range_bias + ((1, 1),)
        else:
            if len(shape_x) < 2:
                raise RuntimeError("bias_add only support shape larger than 2D")
            if not check_equal(shape_x[-1], shape_bias[0]):
                raise RuntimeError("data_format is NHWC, shape_bias must be "
                                   "equal to the last axis of shape_x")
            shape_bias = ()
            range_bias = (())
            for i in range(0, len(shape_x)):
                if i != len(shape_x) - 1:
                    shape_bias = shape_bias + (1,)
                    range_bias = range_bias + ((1, 1),)
                else:
                    shape_bias = shape_bias + (shape_x[-1],)
                    range_bias = range_bias + (range_x[-1],)

    bias["shape"] = shape_bias
    bias["ori_shape"] = shape_bias
    bias["range"] = range_bias

    check_elewise_shape_range([x, bias], support_broadcast=True)

    ins = classify([x, bias], Mode.ELEWISE_WITH_BROADCAST)

    schedules, tensors = [], []
    for (_x, _bias) in ins:
        with tbe_base.compute():
            x_shape, bias_shape = shape_util.variable_shape([_x, _bias], support_broadcast=True)
            x_shape, bias_shape = shape_util.refine_shapes_for_broadcast(x_shape, bias_shape)
            tensor_x = tvm.placeholder(x_shape, dtype_x, "tensor_x")
            tensor_bias = tvm.placeholder(bias_shape, dtype_bias, "tensor_bias")

            res = bias_add_compute(tensor_x, tensor_bias, y, data_format, kernel_name)
            tensors.append((tensor_x, tensor_bias, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
    tbe_base.add_compile_info("_boardcast_bias_shape", shape_bias)
