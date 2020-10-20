# Copyright 2018 Huawei Technologies Co., Ltd
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
leaky_relu
"""
from functools import reduce as reduceIns
from te.utils.op_utils import *

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic


def get_fusion_params(x_tensor, y):
    """
    Get L1 fusion_params
    Parameters
    ----------
    x_tensor : tensor of input data
    y : dict of output data
    Returns
    -------
    fusion_params
    """
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    l1_fusion_type = -1
    if fusion_manager.get_build_cfg() != "disable":
        l1_fusion_type = x_tensor.op.attrs["L1_fusion_type"].value \
            if "L1_fusion_type" in x_tensor.op.attrs else -1
        if l1_fusion_type == 1:
            raise RuntimeError("leaky_relu does not support l1 width fusion")
    is_l1_depth_fusion = l1_fusion_type == 0
    in_l1_flag = x_tensor.op.attrs["addr_type"].value == 1 \
        if "addr_type" in x_tensor.op.attrs else False
    in_valid_shape = x_tensor.op.attrs["valid_shape"] \
        if "valid_shape" in x_tensor.op.attrs else []
    in_slice_offset = x_tensor.op.attrs["slice_offset"] \
        if "slice_offset" in x_tensor.op.attrs else []
    in_select_read_flag = x_tensor.op.tag == "read_select_5d"

    out_l1_flag = False
    out_valid_shape = []
    out_slice_offset = []
    out_select_write_flag = False
    if y is not None:
        out_l1_flag = y.get("addr_type", 0) == 1
        out_valid_shape = y.get("valid_shape", [])
        out_slice_offset = y.get("slice_offset", [])
        out_select_write_flag = bool(out_valid_shape)

    fusion_params = {"is_l1fusion": is_l1_depth_fusion,
                     "l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag,
                     "in_select_read_flag": in_select_read_flag,
                     "in_valid_shape": in_valid_shape,
                     "in_slice_offset": in_slice_offset,
                     "out_l1_flag": out_l1_flag,
                     "out_select_write_flag": out_select_write_flag,
                     "out_valid_shape": out_valid_shape,
                     "out_slice_offset": out_slice_offset}
    return fusion_params


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("leaky_relu")
def leaky_relu_compute(x, y, negative_slope=0, kernel_name="leaky_relu"):
    """
    compute for caffe_relu_layer_cce
    """
    fusion_params = get_fusion_params(x, y)
    res = te.lang.cce.vlrelu(x, negative_slope)
    if x.op.attrs:
        if 'format' in x.op.attrs:
            res.op.attrs['format'] = x.op.attrs['format']
    res.op.attrs["negative_slope"] = negative_slope
    res.op.attrs["ele_fusion_params"] = fusion_params
    res.op.attrs["L1_fusion_type"] = fusion_params["l1_fusion_type"]

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_FLOAT, KERNEL_NAME)
def leaky_relu(x, y, negative_slope=0, kernel_name="leaky_relu"):
    """leaky_relu op for input tensor

       f(x)= x(x>=0) or negative_slope*x(x<0) equal to
       f(x)=negative_slope*x

    Parameters
    ----------
    x : TVM tensor
        input tensor has shape and dtype attributes
    y : dict
        dict with keys(shape and dtype) of output

    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization

    kernel_name : str
        cce kernel name, default value is "leaky_relu"

    Returns
    ------
    None
    """

    # check input tensor shape
    shape = x.get("shape")
    dtype = x.get("dtype")
    check_shape(shape, param_name="x")

    # check input tensor data_type
    check_list = ["float16", "float32", "int32", "int8"]
    check_dtype(dtype.lower(), check_list, param_name="x")
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)
    inp_dtype = dtype.lower()

    l1_fusion_type = -1
    if fusion_manager.get_build_cfg() != "disable":
        l1_fusion_type = x.get("L1_fusion_type", -1)
        if l1_fusion_type == 1:
            raise RuntimeError("leaky_relu does not support l1 width fusion")
    is_l1_depth_fusion = l1_fusion_type == 0
    addr_type = x.get("addr_type", 0)
    valid_shape = x.get("valid_shape", [])
    slice_offset = x.get("slice_offset", [])
    attr_x = {"addr_type": addr_type,
              "valid_shape": valid_shape,
              "slice_offset": slice_offset,
              "L1_fusion_type": l1_fusion_type}

    input_data_x = tvm.placeholder(fuseshape, name="input_data_x",
                                   dtype=inp_dtype, attrs=attr_x)

    with tvm.target.cce():

        res = leaky_relu_compute(input_data_x, y, negative_slope, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data_x, res],
              "l1_fusion_option": is_l1_depth_fusion}
    te.lang.cce.cce_build_code(sch, config)
