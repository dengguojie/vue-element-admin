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
reduce_any_d

  Op_description :
    Reduces `x` along the dimensions given in `axes`. Unless
    `keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
    `axes`. If `keep_dims` is true, the reduced dimensions are
    retained with length 1.

    # reduce_any_d(
    #   x,
    #   y,
    #   axes,
    #   keepdims,
    #   kernel_name='reduce_any_d')

  Supportive_dtype_format :
    ['bool', 'int8']
    ['ND', 'NCHW', 'NHWC']

  Constraint :
    [1] `x` only support 'bool' or 'int8'.
    [2] `aixs` may be int or list, must be in the range `[-rank(x), rank(x))`.
    [3] `keepdims`:An optional `bool`. Defaults to `False`.
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name
# none type
NONE_TYPE = type(None)


@tbe_platform.fusion_manager.fusion_manager.register("reduce_any_d")
def reduce_any_d_compute(x, y, axes, keepdims, kernel_name="reduce_any_d"):
    """
    Parameters
    ----------

    x: input data

    y: shape and dtype of output_res, reserved parameter, not used now

    axes: the first axes to reduce, may be negative to index from the end
           (e.g., -1 for the last axes).
           aixs may be int or list(e.g. [1,2])

    keepdims: if true, retains reduced dimensions with length 1,
               default value is None

    kernel_name: cce kernel name, default value is "reduce_any_d"

    Returns
    -------
    Tensor after any compute

    """
    shape = shape_util.shape_to_list(x.shape)
    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)

    dtype = x.dtype
    data_fp16 = tbe.cast_to(x, "float16")
    data_abs = tbe.vabs(data_fp16)
    res_tmp = tbe.reduce_max(data_abs, axis=axes, keepdims=keepdims)
    res_s8 = tbe.cast_to(res_tmp, dtype, True)
    return res_s8


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_LIST_INT),
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_any_d(x, y, axes, keepdims=None, kernel_name="reduce_any_d"):
    """
    Reduce a tensor on a certain axes based on max

    Parameters:
    ----------
    x : shape and dtype of input_data, only support int8

    y : shape and dtype of output_res, reserved parameter, not used now

    axes : the first axes to reduce, may be negative to index from the end
           (e.g., -1 for the last axes).
           aixs may be int or list(e.g. [1,2])

    keepdims : if true, retains reduced dimensions with length 1,
               default value is None

    kernel_name : cce kernel name, default value is "reduce_any_d"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")

    para_check.check_shape(shape, param_name="x")

    if dtype == "bool":
        dtype = "int8"
    check_list = ("int8", )
    para_check.check_dtype(dtype, check_list, param_name="x")

    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)

    if hasattr(axes, 'index'):
        axes = list(axes)

    axes = shape_util.axis_check(shape_len, axes)

    is_5hdc = para_check.check_and_init_5hdc_reduce_support(x, axes)
    if not is_5hdc:
        shape, axes = shape_util.shape_refine(list(shape), axes)
        shape, axes = shape_util.simplify_axis_shape(shape, axes)

    inp_dtype = dtype.lower()
    data_input = tvm.placeholder(shape, name="data_input_" + kernel_name, dtype=inp_dtype)
    res = reduce_any_d_compute(data_input, y, axes, keepdims, kernel_name)

    if is_5hdc:
        res.ori_shape = x["ori_shape"]
        res.ori_format = x["ori_format"]

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}
    tbe.cce_build_code(sch, config)
