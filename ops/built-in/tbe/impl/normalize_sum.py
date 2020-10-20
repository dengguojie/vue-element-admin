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
normalize_sum
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check

# pylint: disable=locally-disabled,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("normalize_sum")
def normalize_sum_compute(x1, y, data_format, across_spatial=True,
                          kernel_name="normalize_sum"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of x1
    y : dict
        dict of y, include keys(shape and dtype, format)
    data_format: str
        the format of x1
    across_spatial: bool
        indicates whether reduction should cross spatial locations.
        Default(True)
    kernel_name : str
        kernel name, default value is "normalize_sum"

    Returns
    -------
    output tensor
    """

    # set intermediate dtype
    if not tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
        # hisi es, cs
        intermediate_dtype = "float16"
        dtype_cast_mapping = {"int8": "float16"}
    else:
        intermediate_dtype = "float32"
        dtype_cast_mapping = {"int8": "float16", "float16": "float32"}

    x1_cast = x1
    if intermediate_dtype != x1.dtype:
        while x1_cast.dtype in dtype_cast_mapping:
            x1_cast = tbe.cast_to(x1_cast,
                                          dtype_cast_mapping[x1_cast.dtype])

    x1_cast_sqr = tbe.vmul(x1_cast, x1_cast)

    if across_spatial:
        x1_cast_sqr_sum = tbe.sum(x1_cast_sqr, axis=[1, 2, 3],
                                          keepdims=True)
    elif data_format == "NCHW":
        x1_cast_sqr_sum = tbe.sum(x1_cast_sqr, axis=[1], keepdims=True)
    elif data_format == "NHWC":
        x1_cast_sqr_sum = tbe.sum(x1_cast_sqr, axis=[3], keepdims=True)

    return x1_cast_sqr_sum


# pylint: disable=locally-disabled,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, 
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def normalize_sum(x1, y, across_spatial=True, kernel_name="normalize_sum"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype, format of input 1
    y : dict
        shape and dtype, format of output, should be same format as input 1
    across_spatial: bool
        indicates whether reduction should cross spatial locations.
        Default(True)
    kernel_name : str
        kernel name, default value is "normalize_sum"

    Returns
    -------
    None
    """

    shape_1 = x1.get("shape")
    dtype_1 = x1.get("dtype").lower()
    data_format = x1.get("format")

    if len(list(shape_1)) == 2:
        if data_format == "NCHW":
            shape_1 = [shape_1[0], shape_1[1], 1, 1]
        elif data_format == "NHWC":
            shape_1 = [shape_1[0], 1, 1, shape_1[1]]

    para_check.check_shape(shape_1, param_name="x1")

    cce_product = tbe_platform.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        # hisi es, cs
        para_check.check_dtype(dtype_1, ("int8", "float16",), param_name="x1")
    else:
        para_check.check_dtype(dtype_1, ("int8", "float16", "float32",), param_name="x1")

    para_check.check_format(data_format, ("NCHW", "NHWC"), param_name="x1")

    para_check.check_shape(shape_1, min_rank=4, max_rank=4, param_name="x1")

    data_x1 = tvm.placeholder(shape_1, name="data_1", dtype=dtype_1)
    res = normalize_sum_compute(data_x1, y, data_format, across_spatial,
                                kernel_name)

    # pylint: disable=no-member
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "need_build": True,
              "name": kernel_name,
              "tensor_list": [data_x1, res]}

    tbe.cce_build_code(sch, config)