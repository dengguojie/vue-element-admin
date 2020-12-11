# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
conv3d backprop input DSL interface.
"""
import te.lang.cce as tbe
from te.lang.cce.te_compute import util as te_util
import te.platform as tbe_platform
from te.utils import para_check
from te.lang.cce.te_compute import conv3d_backprop_input_general_compute as conv3d_bp_gen_dx
from te import tvm


@para_check.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, (list, tuple),
                             (list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), str, str, dict)
def _conv3d_backprop_input_compute(filters,  # pylint: disable=R0913,R0914
                                  out_backprop, filter_sizes,
                                  input_sizes, strides, padding,
                                  dilations, res_dtype="float16",
                                  kernel_name="conv3d_backprop_input_cce",
                                  group_dict=None):
    """
    DSL interface of conv3d backprop input

    Parameters
    ----------
    filters : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_sizes : shape of weight, [D, H, W, N, C]

    input_sizes : shape of dE/dX, [N, D, H, W, C]

    strides : list of strides, [stridebatch,
                                strided, strideh, stridew, stridechannel]

    padding : list of padding, [pad_front, pad_tail,
                                pad_up, pad_down, pad_left, pad_right]

    dilations : [1, 1, 1, 1, 1] by default

    res_dtype : dE/dX data type, "float16" by default

    kernel_name : name of kernel, "conv3d_backprop_input_cce" by default

    group_dict: the information needed for group convolution, None by default

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    dx_batch, dx_d, dx_h, dx_w, dx_c = input_sizes
    _, _, config_n0 = tbe_platform.CUBE_MKN[res_dtype]['mac']
    shape_dx = (dx_batch, dx_d, te_util.int_ceil_div(dx_c, config_n0), dx_h, dx_w, config_n0)
    pattc = conv3d_bp_gen_dx.DeConvPattern(filter_sizes, strides=strides,
                                    pad=padding, output_shape=shape_dx,
                                    dilations=dilations,
                                    kernel_name=kernel_name,
                                    group_dict=group_dict)
    dy_col = pattc.generate_a(out_backprop)
    w_col = pattc.generate_b(filters)
    dx_ddr = pattc.generate_c(dy_col, w_col)
    return dx_ddr


@para_check.check_input_type(tvm.tensor.Tensor,
                             tvm.tensor.Tensor,
                             (list, tuple),
                             (list, tuple),
                             dict)
def conv3d_dx(filter,
              out_backprop,
              filter_size,
              input_size,
              para_dict):
    """
    DSL interface of conv3d bp dx

    Parameters
    ----------
    filter : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_size : shape of weight, [D, H, W, N, C]

    input_size : shape of dE/dX, [N, D, H, W, C]

    para_dict : dict of parameters
        strides : list of strides, [stridebatch, strided, strideh, stridew, stridechannel]
        pads : list of padding, [pad_front, pad_tail, pad_up, pad_down, pad_left, pad_right]
        dilations : [1, 1, 1, 1, 1] by default
        res_dtype : dE/dX data type, "float16" by default
        kernel_name : conv3d_backprop_input_cce by default
        group_dict : group of parameters

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    strides = para_dict.get("strides")
    pads = para_dict.get("pads")
    dilations = para_dict.get("dilations", [1, 1, 1, 1, 1])
    res_dtype = para_dict.get("res_dtype", "float16")
    kernel_name = para_dict.get("kernel_name", "conv3d_backprop_input_cce")
    group_dict = para_dict.get("group_dict", None)

    return _conv3d_backprop_input_compute(filter,
                                         out_backprop, filter_size,
                                         input_size, strides, pads,
                                         dilations, res_dtype,
                                         kernel_name,
                                         group_dict)
