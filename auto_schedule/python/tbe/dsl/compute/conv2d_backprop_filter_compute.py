#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
conv2d backprop filter DSL interface.
"""
from __future__ import absolute_import
from __future__ import print_function

from functools import reduce

from tbe.common import platform as tbe_platform
from tbe.common import utils as tbe_utils
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.compute import cube_util
from tbe.tvm import api as tvm
from tbe.tvm.expr import Var
from tbe.tvm.tensor import Tensor


# fractal size, only support 16 for now
BLOCK_SIZE = 16

# maximum of int64 (2**63 - 1)
DATA_SIZE_LIMIT_INT64 = 9223372036854775807
INPUTS_H_MAX = 200000
INPUTS_W_MAX = 4096
INPUTS_HW_MIN = 1

# maximum of w in conv1d is (2**31 - 1)
CONV1D_MAX_W = 2147483647

# filterH, filterW must be in [1,255]
FILTER_HW_MAX = 255
FILTER_HW_MIN = 1

# stride must be in [1,63]
STRIDE_HW_MAX = 63
STRIDE_HW_MIN = 1

# dilation must be in [1,255]
DILATION_MIN = 1
DILATION_MAX = 255

# pad must be in [1,255]
PAD_MIN = 0
PAD_MAX = 255

# the bytes length of several dtype
BIT_RATIO_DICT = {
    "int32": 4,
    "float32": 4,
    "float16": 2,
    "uint8": 1,
    "int8": 1,
    "uint4": 0.5,
    "int4": 0.5,
    "bfloat16": 2,
}


def _check_shape_rule(shape, dim, formats, name, allow_zero=False):
    """
    check shape

    """
    if len(shape) != dim:
        dict_args = {}
        dict_args["errCode"] = "E64003"
        dict_args["param_name"] = name
        dict_args["format"] = formats
        dict_args["expect_dim"] = str(dim)
        dict_args["dim"] = str(len(shape))
        error_manager_util.raise_runtime_error(dict_args)
    axis_num = 0
    for dim_x in shape:
        check_zero = False
        if allow_zero:
            check_zero = (dim_x < 0)
            axis_rule = "int and >= 0 "
        else:
            check_zero = (dim_x <= 0)
            axis_rule = "int and > 0"
        if (not isinstance(dim_x, int)) or check_zero:
            dict_args = {}
            dict_args["errCode"] = "E64004"
            dict_args["param_name"] = name
            dict_args["axis_rule"] = axis_rule
            dict_args["wrong_axis"] = str(axis_num)
            dict_args["actual_value"] = str(dim_x)
            error_manager_util.raise_runtime_error(dict_args)
        axis_num = axis_num + 1


def _ceil_div(dividend, divisor):
    """
    do division and round up to an integer

    """
    if divisor == 0:
        dict_args = {}
        dict_args['errCode'] = "E60108"
        dict_args['reason'] = "Division by zero"
        error_manager_util.raise_runtime_error(dict_args)
    return (dividend + divisor - 1) // divisor


def lcm(param1, param2):
    """
    calculate least common multiple
    """
    temp = param1 * param2
    while param1 % param2 != 0:
        param1, param2 = param2, param1 % param2
    return temp // param2


def _check_attr_rule(attr, dim, attr_limit, formats, name):
    """
    check attribute

    """
    attr_min = attr_limit[0]
    attr_max = attr_limit[1]
    if len(attr) != dim:
        dict_args = {}
        dict_args["errCode"] = "E64003"
        dict_args["param_name"] = name
        dict_args["format"] = formats
        dict_args["expect_dim"] = str(dim)
        dict_args["dim"] = str(len(attr))
        error_manager_util.raise_runtime_error(dict_args)
    for attr_x in attr:
        if (not isinstance(attr_x, int)) \
                or attr_x < attr_min or attr_x > attr_max:
            dict_args = {}
            dict_args['errCode'] = "E64001"
            dict_args['range'] = "[{},{}]".format(attr_min, attr_max)
            dict_args['attr_name'] = name
            dict_args["value"] = str(attr_x)
            error_manager_util.raise_runtime_error(dict_args)


def _check_variable_range(variable, minimum, maximum, name):
    """
    check variable range

    """
    if (not isinstance(variable, int)) or variable < minimum \
            or variable > maximum:
        dict_args = {}
        dict_args['errCode'] = "E64001"
        dict_args['range'] = "[{},{}]".format(minimum, maximum)
        dict_args['attr_name'] = name
        dict_args["value"] = str(variable)
        error_manager_util.raise_runtime_error(dict_args)


def _check_addressing_rule(shape, byte_count, limit, name):
    """
    check addressing limit

    """
    # product of all dimension
    product = reduce(lambda x, y: x * y, shape[:])
    if product*byte_count > limit:
        dict_args = {}
        dict_args['errCode'] = "E60020"
        dict_args['attr_name'] = name
        error_manager_util.raise_runtime_error(dict_args)


def check_shape_equal(shape_x, shape_y, param_name1, param_name2):
    """
    check shape dim equal
    """
    if shape_x != shape_y:
        dict_args = {}
        dict_args['errCode'] = "E64002"
        dict_args['param1'] = param_name1
        dict_args['param2'] = param_name2
        dict_args['actual_value'] = "{}, {}".format(shape_x, shape_y)
        error_manager_util.raise_runtime_error(dict_args)


class Conv2dBackpropFilter:
    """
    Conv2dBackpropFilter: compute definition of conv2d_backprop_filter

    Functions
    ----------
    __init__ : initialization

    deconv_dw_input_check_1: parameters check part 1

    deconv_dw_input_check_2: parameters check part 2

    deconv_dw_compute : compute process

    grads_2_matrix : compute definition of loading grads to cbuf

    grads_2_fractal : compute definition of load_2d

    fmap_2_col_matrix : compute definition of set_fmatrix

    fmap_2_col_fractal : compute definition of load_3d

    mad : compute definition of mmad

    """

    def __init__(self, input_x, out_backprop,
                 filter_sizes, strides, padding, dilations, groups,
                 res_dtype="float32",
                 kernel_name="conv2d_backprop_filter_cce"):
        """
        initialization

        Parameters:
        ----------
        input_x : the featuremap data, tvm.placeholder, 5HD shape

        out_backprop : the grads data, tvm.placeholder, 5HD shape

        filter_sizes : 4-D shape, specifies the filter sizes

        strides : 2-D shape in height and width dimension

        padding : 4-D shape in up/down/left/right dimension

        dilations : 4-D shape in batch/channel/height/width dimension

        groups : The number of filter's group. Default value is 1.

        res_dtype : the output data type

        Returns
        -------
        None
        """

        self.fmap, self.grads, self.weight_shape = input_x, out_backprop, list(filter_sizes)
        self.fmap_dtype, self.grads_dtype, self.res_dtype = input_x.dtype, out_backprop.dtype, res_dtype
        self.pad, self.stride, self.dilation = cube_util.shape_to_list(padding), list(strides), list(dilations)
        self.group, self.kernel_name = groups, kernel_name
        self.shapelist, self.group_dict = {}, {}
        self.optag = "conv2d_backprop_filter"

        # 5HD shape
        self.shape_grads_5hd = cube_util.shape_to_list(self.grads.shape)
        self.shape_x_5hd = cube_util.shape_to_list(self.fmap.shape)

        self.shapelist['grads_5hd'] = self.shape_grads_5hd
        self.shapelist['fmap_5hd'] = self.shape_x_5hd

        self.dw_ddr = []
        self.res_tensor = self.dw_ddr  # return tensor of this file to topi

        # flag of special case
        self.flag_all_one_case = False
        self.flag_load3d_special_case = False
        self.conv1d_situation = False
        self.flag_load3d_w_split_case = False
        self.l0b_dma_flag = False

        self.cube_vector_split = tbe_platform_info.get_soc_spec("CUBE_VECTOR_SPLIT")
        self.c0_size = tbe_platform.C0_SIZE

        # for dynamic
        self.dynamic_para = self._get_dynamic_para()
        self.var_map = self.dynamic_para.get("var_map")
        DynamicConv2dBpFilterParams.var_map = self.var_map
        DynamicConv2dBpFilterParams.tiling_info_dict = {
            "op_type": 'conv2d_backprop_filter',
            "A_shape": cube_util.shape_to_list(self.grads.shape),
            "B_shape": cube_util.shape_to_list(self.fmap.shape),
            "C_shape": cube_util.shape_to_list([
                _ceil_div(self.weight_shape[0], BLOCK_SIZE) * BLOCK_SIZE,
                _ceil_div(self.weight_shape[1], BLOCK_SIZE),
                self.weight_shape[2], self.weight_shape[3], BLOCK_SIZE]),
            "A_dtype": self.grads.dtype,
            "B_dtype": self.fmap.dtype,
            "C_dtype": res_dtype,
            "mad_dtype": 'float32',
            "padl": self.pad[2],
            "padr": self.pad[3],
            "padu": self.pad[0],
            "padd": self.pad[1],
            "strideH": self.stride[0],
            "strideW": self.stride[1],
            "strideH_expand": 1,
            "strideW_expand": 1,
            "dilationH": self.dilation[2],
            "dilationW": self.dilation[3],
            "group": 1,
            "bias_flag": 0,
            "fused_double_operand_num": 0,
            "kernel_name": kernel_name,
            "dynamic_shape_flag": True
        }

    def deconv_dw_access(self):
        """
        complete compute generation, including input check,
        compute definition and result record

        """
        if not DynamicConv2dBpFilterParams.is_binary_flag:
            self._deconv_dw_input_check_1()
            self._deconv_dw_input_check_2()
            if not self.var_map:
                self._deconv_dw_input_check_3()
        self._compute_group_dict()
        self.deconv_dw_compute()
        self.res_tensor = self.dw_ddr  # return tensor of this file to topi

    def deconv_dw_compute(self):
        """
        complete compute definition

        """

        fmap_dtype = self.fmap_dtype

        batch_size, grads_channel_1, grads_height, grads_width, grads_c0 = self.shape_grads_5hd
        _, fmap_channel_1, fmap_height, fmap_width, fmap_c0 = self.shape_x_5hd
        _, _, kernel_height, kernel_width = self.weight_shape
        _, _, _, dilation_w = self.dilation

        if not self.var_map and self.flag_load3d_special_case:
            # in this situation, stride_w do no make sense
            # set stride_w be fmap_w_after_pad
            # add kernel_w_after_dilation to pad_right
            # so that grads_w be 2
            self.stride[1] = fmap_width + self.pad[2] + self.pad[3]
            self.pad[3] += (kernel_width - 1) * dilation_w + 1
            grads_width = grads_width * 2
            self.shapelist.get('grads_5hd')[-2] = grads_width
            self.shape_grads_5hd[-2] = grads_width
        if self.var_map and not DynamicConv2dBpFilterParams.is_binary_flag:
            w_one_flag = tvm.var("w_one_flag")
            self.var_map["w_one_flag"] = w_one_flag
            stride_w = tvm.select(w_one_flag == 2,
                                  fmap_width + self.pad[2] + self.pad[3],
                                  self.stride[1])
            self.stride[1] = stride_w

            pad_r = tvm.select(w_one_flag == 2,
                               self.pad[3] + (kernel_width - 1) * dilation_w + 1,
                               self.pad[3])
            self.pad[3] = pad_r
            grads_width = grads_width * w_one_flag
            self.shapelist.get('grads_5hd')[-2] = grads_width
            self.shape_grads_5hd[-2] = grads_width

        # group dict
        group_dict = self.group_dict
        real_g = group_dict.get("real_g")
        fmap_c1_g = group_dict.get("cin1_g")
        grads_channel_g = group_dict.get("cout_g")

        # align to 16
        hw_mad = (grads_height * grads_width + BLOCK_SIZE - 1) \
                 // BLOCK_SIZE * BLOCK_SIZE
        hw_mad_1 = (grads_height * grads_width + BLOCK_SIZE - 1) \
                 // BLOCK_SIZE
        wo_mad_1 = (grads_width + BLOCK_SIZE - 1) // BLOCK_SIZE

        # move grads to L1
        grads_shape_matrix = (batch_size,
                              grads_channel_1,
                              grads_height * grads_width,
                              grads_c0)

        if self.grads_dtype == "float32":
            grads_shape_matrix = (
                batch_size,
                hw_mad_1,
                grads_channel_1,
                BLOCK_SIZE,
                grads_c0
            )

        if self.flag_load3d_w_split_case:
            grads_shape_matrix = (batch_size,
                                  grads_channel_1,
                                  grads_height,
                                  grads_width,
                                  grads_c0)
            if self.grads_dtype == "float32":
                grads_shape_matrix = (batch_size,
                                      grads_height,
                                      wo_mad_1,
                                      grads_channel_1,
                                      BLOCK_SIZE,
                                      grads_c0)

        self.shapelist['grads_matrix'] = grads_shape_matrix

        grads_matrix = self._grads_2_matrix(grads_shape_matrix, self.grads)

        # move grads_matrix to L0A and do transpose
        grads_shape_fractal = (real_g,
                               batch_size,
                               grads_channel_g // grads_c0,
                               hw_mad_1,
                               grads_c0,
                               BLOCK_SIZE)

        if self.grads_dtype == "float32":
            # transpose to make sure k0 axis in mad is 8
            grads_shape_fractal = (
                real_g,
                batch_size,
                grads_channel_g // grads_c0 // 2,
                hw_mad_1 * 2,
                grads_c0 * 2,
                BLOCK_SIZE // 2
            )

        if self.flag_load3d_w_split_case:
            grads_shape_fractal = (real_g,
                                   batch_size,
                                   grads_channel_g // grads_c0,
                                   grads_height,
                                   wo_mad_1,
                                   grads_c0,
                                   BLOCK_SIZE)
            if self.grads_dtype == "float32":
                grads_shape_fractal = (
                    real_g,
                    batch_size,
                    grads_channel_g // grads_c0 // 2,
                    grads_height,
                    wo_mad_1 * 2,
                    grads_c0 * 2,
                    BLOCK_SIZE // 2
                )

        self.shapelist['grads_fractal'] = grads_shape_fractal
        grads_fractal = self._grads_2_fractal(grads_shape_fractal, grads_matrix)

        if not self.flag_all_one_case:
            if not self.var_map:
                # Load 3D Data Flow:
                # fmap in DDR to fmap_matrix in L1 to fmap_fractal_nZ in L0B

                # Dma Mode Data Flow:
                # fmap in DDR to fmap_ub_pad in UB to fmap_matrix in L1 to
                # fmap_fractal_before_zZ in L1 to fmap_fractal_nZ in L0B
                fmap_ub = None
                if self.l0b_dma_flag and self.pad != [0, 0, 0, 0]:
                    pad_top, pad_bottom, pad_left, pad_right = self.pad
                    fmap_ub_shape = (batch_size,
                                     fmap_channel_1,
                                     fmap_height + pad_top + pad_bottom,
                                     fmap_width + pad_left + pad_right,
                                     fmap_c0)
                    fmap_ub = self._fmap_2_ub(fmap_ub_shape, fmap_height, fmap_width)

                # shape of fmap_original_matrix, corresponding to set_fmatrix
                fmap_shape_original_matrix = (batch_size,
                                              grads_height*grads_width,
                                              fmap_channel_1,
                                              kernel_height,
                                              kernel_width,
                                              fmap_c0)

                if self.flag_load3d_w_split_case:
                    fmap_shape_original_matrix = (batch_size,
                                                  grads_height,
                                                  grads_width,
                                                  fmap_channel_1,
                                                  kernel_height,
                                                  kernel_width,
                                                  fmap_c0)

                self.shapelist['fmap_original_matrix'] = fmap_shape_original_matrix

                if fmap_ub is None:
                    fmap_l1_before = self.fmap
                else:
                    fmap_l1_before = fmap_ub
                fmap_matrix = self._fmap_2_matrix(fmap_shape_original_matrix,
                                                  fmap_l1_before, fmap_dtype)
                # load 3d: move fmap to L0B
                # dma mode: change to zZ in L1 first
                fmap_shape_fmap_matrix = (real_g,
                                          batch_size,
                                          hw_mad_1,
                                          fmap_c1_g * kernel_height * kernel_width,
                                          fmap_c0,
                                          BLOCK_SIZE)
                if self.fmap_dtype == "float32":
                    fmap_shape_fmap_matrix = (
                        real_g,
                        batch_size,
                        hw_mad_1 * 2,
                        fmap_c1_g * kernel_height * kernel_width // 2,
                        fmap_c0 * 2,
                        BLOCK_SIZE // 2
                    )

                if self.flag_load3d_w_split_case:
                    fmap_shape_fmap_matrix = (real_g,
                                              batch_size,
                                              grads_height,
                                              wo_mad_1,
                                              fmap_c1_g * kernel_height * kernel_width,
                                              fmap_c0,
                                              BLOCK_SIZE)
                    if self.fmap_dtype == "float32":
                        fmap_shape_fmap_matrix = (real_g,
                                                  batch_size,
                                                  grads_height,
                                                  wo_mad_1 * 2,
                                                  fmap_c1_g * kernel_height * kernel_width // 2,
                                                  fmap_c0 * 2,
                                                  BLOCK_SIZE // 2)

                self.shapelist['fmap_fmap_matrix'] = fmap_shape_fmap_matrix

                if self.l0b_dma_flag:
                    fmap_shape_fmap_matrix = (
                        real_g,
                        batch_size,
                        hw_mad_1,
                        fmap_c1_g*kernel_height*kernel_width,
                        BLOCK_SIZE,
                        fmap_c0
                    )
                fmap_fractal = self._fmap_2_fractal(fmap_shape_fmap_matrix, fmap_matrix, fmap_dtype)

                # dma_mode: move fmap(zZ) to L0B
                # swap the last two axes
                if self.l0b_dma_flag:
                    fmap_shape_fmap_matrix = self.shapelist.get('fmap_fmap_matrix')
                    fmap_fractal_with_dma = tvm.compute(fmap_shape_fmap_matrix,
                                                        lambda *indices:
                                                        fmap_fractal(*indices[:-2], indices[-1], indices[-2]),
                                                        name='fmap_2_fractal_dma',
                                                        tag='fmap_2_fractal_dma')
                    fmap_fractal = fmap_fractal_with_dma
            else:
                fmap_l1_shape = (real_g, batch_size, fmap_c1_g,
                                    fmap_height, fmap_width, fmap_c0)
                fmap_l1 = tvm.compute(fmap_l1_shape,
                                        lambda g, n, c1, h, w, c0:
                                        self.fmap(n, c1 + g * fmap_c1_g, h, w, c0),
                                        name='dw_fmap_l1', tag='dw_fmap_l1',
                                        attrs={'group_dict':self.group_dict})

                fmap_shape_fmap_matrix = (real_g,
                                          batch_size,
                                          hw_mad_1,
                                          fmap_c1_g*kernel_height*kernel_width,
                                          fmap_c0,
                                          BLOCK_SIZE)
                img2col_para = (fmap_l1, kernel_height, kernel_width,
                            self.pad, self.stride, grads_width, self.dilation)
                fmap_fractal = self._im2col_fractal_v2(fmap_shape_fmap_matrix,
                                                       img2col_para)

        # else: all_one_case, using load_2d instead of load_3d
        else:
            # shape of fmap_matrix
            fmap_shape_matrix = (batch_size,
                                 fmap_channel_1,
                                 fmap_height*fmap_width,
                                 fmap_c0)

            self.shapelist['fmap_matrix'] = fmap_shape_matrix

            fmap_matrix = self._fmap_2_matrix_load2d(fmap_shape_matrix,
                                                     self.fmap)

            # move fmap to L0B
            fmap_shape_fractal = (real_g,
                                  batch_size,
                                  hw_mad//BLOCK_SIZE,
                                  fmap_c1_g*kernel_height*kernel_width,
                                  fmap_c0,
                                  BLOCK_SIZE)
            self.shapelist['fmap_matrix'] = fmap_shape_fractal

            fmap_fractal = self._fmap_2_fractal_load2d(fmap_shape_fractal,
                                                       fmap_matrix)

        # shape of result dw [group,n1,m,n0]
        if self.fmap_dtype == "float32" and self.grads_dtype == "float32":
            dw_shape = (
                real_g,
                fmap_c1_g // 2 * kernel_height * kernel_width,
                grads_channel_g,
                fmap_c0 * 2
            )
            dw_cc = self._mad(dw_shape, grads_fractal, fmap_fractal)
            # do channel split
            dw_c_split_shape = (real_g, fmap_c1_g, kernel_height*kernel_width,
                                grads_channel_g, fmap_c0)
            khkw = kernel_height * kernel_width
            dw_c_split = tvm.compute(dw_c_split_shape,
                                     lambda g_idx, c1_idx, kk_idx, grads_c_idx, c0_idx:
                                     dw_cc(
                                         g_idx,
                                         c1_idx // 2 * khkw + kk_idx,
                                         grads_c_idx,
                                         c1_idx % 2 * self.c0_size + c0_idx
                                     ).astype(self.res_dtype),
                                     name='dw_c_split', tag=self.optag + "_c_split",
                                     attrs={'kernel_name': self.kernel_name})
            self.dw_ddr = dw_c_split
        else:
            dw_shape = (real_g, fmap_c1_g*kernel_height*kernel_width,
                        grads_channel_g, fmap_c0)
            self.shapelist['dw'] = dw_shape
            dw_cc = self._mad(dw_shape, grads_fractal, fmap_fractal)
            self.dw_ddr = dw_cc

        return 1

    def _get_dynamic_para(self):
        n_dim = 0
        h_dim = 2
        w_dim = 3
        fmap_range = []
        var_map = {}

        if isinstance(self.fmap.shape[n_dim], Var):
            fmap_range.append(get_te_var("batch").get_bound())
            var_map["batch"] = get_te_var("batch").get_tvm_var()
        else:
            fmap_range.append((self.fmap.shape[n_dim], self.fmap.shape[n_dim]))

        if isinstance(self.fmap.shape[h_dim], Var):
            fmap_range.append(get_te_var("fmap_h").get_bound())
            var_map["fmap_h"] = get_te_var("fmap_h").get_tvm_var()
            var_map["dedy_h"] = get_te_var("dedy_h").get_tvm_var()
        else:
            fmap_range.append((self.fmap.shape[h_dim], self.fmap.shape[h_dim]))

        if isinstance(self.fmap.shape[w_dim], Var):
            fmap_range.append(get_te_var("fmap_w").get_bound())
            var_map["fmap_w"] = get_te_var("fmap_w").get_tvm_var()
            var_map["dedy_w"] = get_te_var("dedy_w").get_tvm_var()
        else:
            fmap_range.append((self.fmap.shape[w_dim], self.fmap.shape[w_dim]))

        dynamic_para = {
            "fmap_range": fmap_range,
            "var_map": var_map,
        }
        return dynamic_para

    def _deconv_dw_input_check_1(self):
        """
        do input parameters check part1

        """
        # check of data type
        if self.fmap_dtype != self.grads_dtype:
            dict_args = {}
            dict_args["errCode"] = "E60038"
            dict_args["desc"] = "The fmap data type is not same as the out_backprop data type."
            error_manager_util.raise_runtime_error(dict_args)
        # update c0 size
        self.c0_size = tbe_platform.CUBE_MKN.get(self.fmap_dtype).get("mac")[1]
        if not tbe_platform.intrinsic_check_support("Intrinsic_mmad", "f162f32") and \
                self.res_dtype != "float16":
            dict_args = {}
            dict_args["errCode"] = "E60005"
            dict_args["param_name"] = "y"
            dict_args["expected_dtype_list"] = "float16 for lhisi"
            dict_args["dtype"] = self.res_dtype
            error_manager_util.raise_runtime_error(dict_args)
        if tbe_platform.intrinsic_check_support("Intrinsic_mmad", "f162f32") and \
                self.res_dtype != "float32":
            dict_args = {}
            dict_args["errCode"] = "E60005"
            dict_args["param_name"] = "y"
            dict_args["expected_dtype_list"] = "float32"
            dict_args["dtype"] = self.res_dtype
            error_manager_util.raise_runtime_error(dict_args)

        # check shape
        # each element must be positive int

        _check_shape_rule(self.weight_shape, 4, "NCHW", "filter_sizes")
        _check_shape_rule(self.stride, 2, "height_weight", "stride")
        _check_shape_rule(self.dilation, 4, "NCHW", "dilation")

        _, _, kernel_height, kernel_width = self.weight_shape

        # pad: pad_top, pad_bottom, pad_left, pad_right
        fmap_height_after_pad = self.shape_x_5hd[2] + self.pad[0] + self.pad[1]

        weight_shape_n = _ceil_div(self.weight_shape[0], self.c0_size) * self.c0_size
        weight_shape_c = _ceil_div(self.group * self.weight_shape[1], self.c0_size) * self.c0_size
        check_shape_equal(self.shape_grads_5hd[4], self.c0_size, "grads_c0", "c0_size")
        check_shape_equal(self.shape_x_5hd[4], self.c0_size, "fmap_c0", "c0_size")
        check_shape_equal(self.shape_grads_5hd[1] * self.shape_grads_5hd[4], \
            weight_shape_n, "grads's C1*C0", "weight's N")
        check_shape_equal(self.shape_x_5hd[1] * self.shape_x_5hd[4], \
            weight_shape_c, "fmap's C1*C0", "weight's C")

        if self.shape_grads_5hd[0] != self.shape_x_5hd[0]:
            dict_args = {}
            dict_args['errCode'] = "E64002"
            dict_args['param1'] = "x's N"
            dict_args['param2'] = "out_backprop's N"
            dict_args['actual_value'] = "{}, {}".\
                format(self.shape_x_5hd[0], self.shape_grads_5hd[0])
            error_manager_util.raise_runtime_error(dict_args)

        def _check_attr_range(attr_list, lower_limit, upper_limit):
            for attr in attr_list:
                if not isinstance(attr, int) or attr < lower_limit or attr > upper_limit:
                    return False
            return True

        # special supporting for a unique case, there are 2 conditions:
        # (1) height & weight of x/output_backprop/filter are all 1
        # (2) strides is [1,1]
        def _set_all_one_case_flag():
            if not self.var_map and self.stride == [1, 1] and self.shape_x_5hd[2:4] == [1, 1] \
                    and self.shape_grads_5hd[2:4] == [1, 1] \
                    and self.weight_shape[2:4] == [1, 1]:
                self.flag_all_one_case = True
            DynamicConv2dBpFilterParams.flag_all_one_case = \
                                                    self.flag_all_one_case

        def _set_load3d_special_case_flag():
            # limitation by chip:
            # load3d instruction not support out_w = 1
            # only Ascend310 and Hi3796CS can support
            if tbe_platform_info.get_soc_spec("SHORT_SOC_VERSION") not in ["Ascend310", "Hi3796CV300CS", "SD3403"] \
                    and self.shape_grads_5hd[2] != 1 \
                    and self.shape_grads_5hd[3] == 1:
                self.flag_load3d_special_case = True
            else:
                self.flag_load3d_special_case = False

        # conv1d situation, all params in h is 1
        # support w be in [1,2**31 - 1]
        def _set_conv1d_situation_flag():
            if fmap_height_after_pad == 1 and kernel_height == 1 \
                    and self.stride[0] == 1 and self.dilation[2] == 1:
                self.conv1d_situation = True

        def _set_load3d_w_split_case_flag():
            """
            Helper function for load3d w split check.

            """
            _, stride_w = self.stride
            _, _, dilation_h, dilation_w = self.dilation
            kernel_h_dilation = (kernel_height - 1) * dilation_h + 1
            kernel_w_dilation = (kernel_width - 1) * dilation_w + 1
            input_dtype_size = BIT_RATIO_DICT.get(self.fmap_dtype, 2)
            l1_size = tbe_platform.get_soc_spec("L1_SIZE")
            al1_min_byte = BLOCK_SIZE * BLOCK_SIZE * input_dtype_size
            # the min load size of Ho is kernel_h
            bl1_min_byte = kernel_h_dilation * (
                (BLOCK_SIZE - 1) * stride_w + kernel_w_dilation) * BLOCK_SIZE * input_dtype_size
            conv1d_flag = fmap_height_after_pad == 1 and kernel_height == 1 and self.stride[0] == 1

            w_split_support_flag = ((not self.var_map)
                                    and (not conv1d_flag)
                                    and (not self.flag_load3d_special_case)
                                    and (al1_min_byte + bl1_min_byte) < l1_size
                                    and _check_attr_range([kernel_height, kernel_width], FILTER_HW_MIN, FILTER_HW_MAX)
                                    and _check_attr_range(self.stride, STRIDE_HW_MIN, STRIDE_HW_MAX)
                                    and _check_attr_range(self.pad, PAD_MIN, PAD_MAX))

            if w_split_support_flag and (self.shape_x_5hd[3] > INPUTS_W_MAX or not _min_l1_check()):
                self.flag_load3d_w_split_case = True

        def _set_dma_flag(attr, attr_limit):
            for attr_value in attr:
                if attr_value > attr_limit:
                    self.l0b_dma_flag = True

        def _min_l1_check():
            """
            L1 limitation, Mainly required by chip
            """
            stride_h, stride_w = self.stride
            _, _, dilation_h, dilation_w = self.dilation
            kernel_h_dilation = (kernel_height - 1) * dilation_h + 1
            kernel_w_dilation = (kernel_width - 1) * dilation_w + 1
            width_grads = self.shape_grads_5hd[3]
            input_dtype_size = BIT_RATIO_DICT.get(self.fmap_dtype, 2)

            al1_min_byte = BLOCK_SIZE * BLOCK_SIZE * input_dtype_size

            if self.conv1d_situation:
                kl1_min = (BLOCK_SIZE - 1) * stride_w + kernel_w_dilation
            else:
                kl1_min = self.shape_x_5hd[3]

            if width_grads >= BLOCK_SIZE:
                if width_grads % BLOCK_SIZE == 0:
                    bl1_min_byte = kernel_h_dilation * kl1_min * BLOCK_SIZE * input_dtype_size
                else:
                    bl1_min_byte = (kernel_h_dilation + stride_h) * kl1_min * BLOCK_SIZE * input_dtype_size
            else:
                bl1_align_factor = _ceil_div(BLOCK_SIZE, width_grads)
                if BLOCK_SIZE % width_grads == 0:
                    bl1_min_byte = (kernel_h_dilation + (bl1_align_factor - 1) * stride_h) * kl1_min * \
                                   BLOCK_SIZE * input_dtype_size
                else:
                    bl1_min_byte = (kernel_h_dilation + bl1_align_factor * stride_h) * kl1_min * \
                                   BLOCK_SIZE * input_dtype_size
            l1_size = tbe_platform.get_soc_spec("L1_SIZE")

            return (al1_min_byte + bl1_min_byte) <= l1_size

        def _check_dma_mode():
            """
            over-spec scene, replace dma_copy with load_3d
            support scene:
                1. chip have ub buffer
                2. not transdata fusion
                3. dilation is [1, 1, 1, 1]
                4. w cannot split
            ----------
            following scene will choose dma mode:
                1. over l1 buffer
                2. kernel > 255
                3. stride > 63
                4. pad > 255
                5. w > 4096
                6. load3d_special_case and w > 63
            """
            is_dma_scene = ((not self.var_map)
                            and (not self.cube_vector_split)
                            and (self.fmap.op.tag != "NHWC_trans_5HD")
                            and self.dilation == [1, 1, 1, 1]
                            and (not self.flag_load3d_w_split_case))
            if is_dma_scene:
                is_special_case = self.flag_load3d_special_case and self.shape_x_5hd[3] > STRIDE_HW_MAX
                if (not _min_l1_check()) or is_special_case:
                    self.l0b_dma_flag = True
                if not self.conv1d_situation:
                    _set_dma_flag([self.shape_x_5hd[3], self.shape_grads_5hd[3]], INPUTS_W_MAX)
                _set_dma_flag([kernel_height, kernel_width], 255)
                _set_dma_flag(self.stride, STRIDE_HW_MAX)
                _set_dma_flag(self.pad, 255)
            elif not self.flag_load3d_w_split_case and not _min_l1_check():
                dict_args = {}
                dict_args["errCode"] = "E60026"
                error_manager_util.raise_runtime_error(dict_args)

        _set_all_one_case_flag()
        _set_load3d_special_case_flag()
        _set_conv1d_situation_flag()
        if not self.var_map:
            _set_load3d_w_split_case_flag()
            _check_dma_mode()

        if not self.l0b_dma_flag:
            _check_variable_range(kernel_height, 1, 255, "height of filter")
            _check_variable_range(kernel_width, 1, 255, "width of filter")
            _check_attr_rule(self.stride, 2, [1, 63],
                            "[strideH, strideW]", "stride")
        return True

    def _deconv_dw_input_check_2(self):
        """
        do input parameters check part2

        """
        stride_height, stride_width = self.stride
        pad_top, pad_bottom, pad_left, pad_right = self.pad
        _, grads_channel_1, grads_height, grads_width, grads_c0 = self.shape_grads_5hd
        _, fmap_channel_1, fmap_height, fmap_width, fmap_c0 = self.shape_x_5hd
        kernel_batch, _, kernel_height, kernel_width = self.weight_shape
        dilationn, dilationc, dilationh, dilationw = self.dilation

        def _check_dilation():
            dilation_min, dilation_max = 1, 255
            if dilationn != 1 or dilationc != 1:
                dict_args = {}
                dict_args["errCode"] = "E60023"
                dict_args["dilation_n"] = str(dilationn)
                dict_args["dilation_c"] = str(dilationc)
                error_manager_util.raise_runtime_error(dict_args)
            _check_variable_range(dilationh, dilation_min, dilation_max, "dilationh")
            _check_variable_range(dilationw, dilation_min, dilation_max, "dilationw")

        def _check_groups():
            if kernel_batch % self.group != 0:
                dict_args = {
                    "errCode": "E60108",
                    "reason": "outbackprop's channel must be a multiple of groups"
                }
                error_manager_util.raise_runtime_error(dict_args)

        _check_dilation()
        _check_groups()

        # coupling range check
        if not isinstance(fmap_height, Var):
            dilation_kernel_height = (kernel_height - 1) * dilationh + 1
            computed_grads_height = (fmap_height - dilation_kernel_height + \
                                    pad_top + pad_bottom)//stride_height + 1
            if computed_grads_height != grads_height:
                dict_args = {}
                dict_args["errCode"] = "E60024"
                error_manager_util.raise_runtime_error(dict_args)
            fmap_height_after_pad = fmap_height + pad_top + pad_bottom
            if dilation_kernel_height > fmap_height_after_pad:
                dict_args = {}
                dict_args["errCode"] = "E60014"
                dict_args["h_of_x"] = str(fmap_height_after_pad)
                dict_args["h_of_filter"] = str(dilation_kernel_height)
                error_manager_util.raise_runtime_error(dict_args)

        if not isinstance(fmap_width, Var):
            dilation_kernel_width = (kernel_width - 1) * dilationw + 1
            computed_grads_width = (fmap_width - dilation_kernel_width + \
                                    pad_left + pad_right)//stride_width + 1
            if computed_grads_width != grads_width:
                dict_args = {}
                dict_args["errCode"] = "E60025"
                error_manager_util.raise_runtime_error(dict_args)
            fmap_width_after_pad = fmap_width + pad_left + pad_right
            if dilation_kernel_width > fmap_width_after_pad:
                dict_args = {}
                dict_args["errCode"] = "E60015"
                dict_args["w_of_x"] = str(fmap_width_after_pad)
                dict_args["w_of_filter"] = str(dilation_kernel_width)
                error_manager_util.raise_runtime_error(dict_args)

        if not self.var_map:
            _check_addressing_rule(self.shape_grads_5hd, 2, DATA_SIZE_LIMIT_INT64, 'shape_grads_5hd')
            _check_addressing_rule(self.shape_x_5hd, 2, DATA_SIZE_LIMIT_INT64, 'shape_x_5hd')
        # int64 addressing limit of tvm
        kernel_fractal = (fmap_channel_1*kernel_height*kernel_width,
                          grads_channel_1*grads_c0,
                          fmap_c0)

        # because of atomic write, dw_ubuf does not tiled, cannot exceed int32
        # limit (change to int64 limit after tvm v0.6)

        _check_addressing_rule(kernel_fractal, 4, DATA_SIZE_LIMIT_INT64, 'kernel_fractal')
        return True

    def _deconv_dw_input_check_3(self):
        # check shape
        # each element must be positive int
        _check_shape_rule(self.shape_grads_5hd, 5, "NC1HWC0", "out_backprop")
        _check_shape_rule(self.shape_x_5hd, 5, "NC1HWC0", "x")
        _check_shape_rule(self.pad, 4, "up_bottom_left_right", "pad",
                        allow_zero=True)

        # individual range check
        inputs_h_max = INPUTS_H_MAX
        inputs_w_max = INPUTS_W_MAX
        _, _, grads_height, grads_width, _ \
            = self.shape_grads_5hd

        if self.conv1d_situation or self.l0b_dma_flag or self.flag_load3d_w_split_case:
            inputs_w_max = CONV1D_MAX_W

        _check_variable_range(grads_height, INPUTS_HW_MIN,
                              inputs_h_max, "height of out_backprop")
        _check_variable_range(grads_width, INPUTS_HW_MIN,
                              inputs_w_max, "width of out_backprop")
        _check_variable_range(self.shape_x_5hd[2], INPUTS_HW_MIN,
                              inputs_h_max, "height of x")
        _check_variable_range(self.shape_x_5hd[3], INPUTS_HW_MIN,
                              inputs_w_max, "width of x")

        # limitation by chip:
        # if only fmap w after padding equals to filter w after dilation
        # and soc_version is Ascend910
        # then only support fmap w not larger than STRIDE_HW_MAX now
        if self.flag_load3d_special_case and not self.l0b_dma_flag:
            _check_variable_range(self.shape_x_5hd[3], INPUTS_HW_MIN,
                                  STRIDE_HW_MAX,
                                  "width of x")

        if isinstance(self.pad[0], int) and not self.l0b_dma_flag:
            _check_attr_rule(self.pad, 4, [0, 255], \
                             "[up,down,left,right]", "pad")
        return True

    def _compute_group_dict(self):
        """
        calculate the params of group convlution

        """
        groups = self.group
        cout, cin, _, _ = self.weight_shape
        fmap_c = cin * groups
        c0_size = self.c0_size
        if not DynamicConv2dBpFilterParams.is_binary_flag:
            mag_factor0 = lcm(fmap_c // groups, BLOCK_SIZE) // (fmap_c // groups)
            mag_factor1 = lcm(cout // groups, BLOCK_SIZE) // (cout // groups)
            mag_factor = min(lcm(mag_factor0, mag_factor1), groups)

            cin_g = (mag_factor * fmap_c // groups + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
            cin1_g = cin_g // c0_size
            cout_g = (mag_factor * cout // groups + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

            group_dict = {
                "real_g": (groups + mag_factor - 1) // mag_factor,
                "mag_factor": mag_factor,
                "cin1_g": cin1_g,
                "cout_g": cout_g,
                "cin_ori": fmap_c,
                "cout_ori": cout
            }
        else:
            group_dict = {
                "real_g": groups,
                "mag_factor": 1,
                "cin1_g": get_te_var("fmap_c1").get_tvm_var(),
                "cout_g": get_te_var("dedy_c1").get_tvm_var() * self.c0_size,
                "cin_ori": fmap_c,
                "cout_ori": cout
            }
        tiling_info_dict_tmp = DynamicConv2dBpFilterParams.tiling_info_dict
        tiling_info_dict_tmp["group"] = group_dict.get("real_g")
        tiling_info_dict_tmp.get("A_shape")[1] = group_dict.get("cout_g") // c0_size
        tiling_info_dict_tmp.get("B_shape")[1] = group_dict.get("cin1_g")
        tiling_info_dict_tmp.get("C_shape")[0] = group_dict.get("cout_g")
        tiling_info_dict_tmp.get("C_shape")[1] = group_dict.get("cin1_g")
        DynamicConv2dBpFilterParams.tiling_info_dict = tiling_info_dict_tmp
        self.group_dict = group_dict

    def _grads_2_matrix(self, grads_shape_matrix, grads):
        """
        compute definiton of loading grads to L1

        Parameters:
        ----------
        grads_shape_matrix : shape of result tensor in L1

        grads : input tensor in ddr

        Returns
        -------
        None
        """
        def __grads_2_matrix_compute(indices, grads):
            """
            do coordinate calculation
            """
            if self.flag_load3d_w_split_case:
                return grads(*indices)

            grads_width = self.shapelist.get('grads_5hd')[3]
            batch_indices, grads_c1_indices, hw_mad_indices, grads_c0_indices = indices

            # calculate index of grads according to indice of grads_matrix
            batch_size_index = batch_indices
            grads_c1_index = grads_c1_indices
            grads_height_index = (hw_mad_indices // grads_width)
            grads_width_index = (hw_mad_indices % grads_width)
            grads_c0_index = grads_c0_indices

            if not self.var_map and self.flag_load3d_special_case:
                # make sure the index won't exceed real grads_w
                grads_width_index = (hw_mad_indices % (grads_width // 2))
            if self.var_map and not DynamicConv2dBpFilterParams.is_binary_flag:
                w_one_flag = self.var_map["w_one_flag"]
                grads_width_index = (hw_mad_indices % (grads_width // w_one_flag))

            return grads(batch_size_index, grads_c1_index,
                         grads_height_index, grads_width_index, grads_c0_index)

        def __grads_2_zz_matrix_compute(indices, grads):
            """
            do coordinate calculation for fp32
            """
            grads_w = self.shapelist.get('grads_5hd')[3]
            if self.flag_load3d_w_split_case:
                (batch_indices, ho_indices, wo_mad_1_indices, grads_c1_indices, wo_mad_0_indices,
                grads_c0_indices) = indices
                grads_h_index = ho_indices
                wo_mad_indices = wo_mad_1_indices * BLOCK_SIZE + wo_mad_0_indices
                grads_w_index = wo_mad_indices % grads_w
            else:
                (batch_indices, hw_mad_1_indices, grads_c1_indices, hw_mad_0_indices, grads_c0_indices) = indices
                hw_mad_indices = hw_mad_1_indices * BLOCK_SIZE + hw_mad_0_indices
                grads_h_index = hw_mad_indices // grads_w
                grads_w_index = hw_mad_indices % grads_w
            return grads(
                    batch_indices,
                    grads_c1_indices,
                    grads_h_index,
                    grads_w_index,
                    grads_c0_indices
                )

        func_name = __grads_2_zz_matrix_compute if self.grads_dtype == "float32" else __grads_2_matrix_compute
        return tvm.compute(grads_shape_matrix,
                           lambda *indices:
                           func_name(indices, grads),
                           name='grads_2_matrix',
                           tag='grads_2_matrix')

    def _grads_2_fractal(self, grads_shape_fractal, grads_2_matrix):
        """
        compute definiton of loading grads_matrix to L0A

        Parameters:
        ----------
        grads_shape_fractal : shape of result tensor in L0A

        grads_2_matrix : input tensor in L1

        Returns
        -------
        None
        """

        def __grads_2_fractal_w_split_compute(indices, grads_2_matrix):
            """
            Compute function for w-split scene.
            """
            (group_indices, batch_indices,
            grads_c1_indices, ho_indices, wo_mad_1_indices,
            grads_c0_indices, wo_mad_0_indices) = indices
            if self.grads_dtype == "float32":
                # hw axis in matrix is 16 aligned, while in fractal is 8 aligned
                wo_mad_index = wo_mad_1_indices * self.c0_size + wo_mad_0_indices
                wo_mad_1_index = wo_mad_index // BLOCK_SIZE
                wo_mad_0_index = wo_mad_index % BLOCK_SIZE

                # c axis in matrix is 8 aligned, while in fractal is 16 aligned
                grads_c_index = group_indices * self.group_dict.get("cout_g") \
                                + grads_c1_indices * BLOCK_SIZE + grads_c0_indices
                grads_c1_index = grads_c_index // self.c0_size
                grads_c0_index = grads_c_index % self.c0_size

                return grads_2_matrix(batch_indices, ho_indices,
                                      wo_mad_1_index, grads_c1_index,
                                      wo_mad_0_index, grads_c0_index)
            else:
                grads_c1_index = (group_indices * self.group_dict.get("cout_g")) // BLOCK_SIZE + grads_c1_indices
                grads_w_index = wo_mad_1_indices * BLOCK_SIZE + wo_mad_0_indices
                grads_c0_index = grads_c0_indices
                return grads_2_matrix(batch_indices, grads_c1_index,
                                      ho_indices, grads_w_index, grads_c0_index)

        def __grads_2_fractal_compute(indices, grads_2_matrix):
            """
            do coordinate calculation
            """
            (group_indices, batch_indices,
            grads_c1_indices, hw_mad_1_indices,
            grads_c0_indices, hw_mad_0_indices) = indices

            batch_size_index = batch_indices
            grads_c1_index = (
                                group_indices * self.group_dict.get("cout_g")
                                ) // BLOCK_SIZE + grads_c1_indices
            grads_hw_index = (
                                hw_mad_1_indices * BLOCK_SIZE
                                ) + hw_mad_0_indices
            grads_c0_index = grads_c0_indices

            return grads_2_matrix(batch_size_index, grads_c1_index,
                                    grads_hw_index, grads_c0_index)

        def __grads_2_zz_fractal_compute(indices, grads_2_matrix):
            """
            do coordinate calculation for fp32
            """
            (group_indices, batch_indices,
            grads_c1_indices, hw_mad_1_indices,
            grads_c0_indices, hw_mad_0_indices) = indices

            # hw axis in matrix is 16 aligned, while in fractal is 8 aligned
            hw_mad_index = hw_mad_1_indices * self.c0_size + hw_mad_0_indices
            hw_mad_1_index = hw_mad_index // BLOCK_SIZE
            hw_mad_0_index = hw_mad_index % BLOCK_SIZE

            # c axis in matrix is 8 aligned, while in fractal is 16 aligned
            grads_c_index = group_indices * self.group_dict.get("cout_g") \
                            + grads_c1_indices * BLOCK_SIZE + grads_c0_indices
            grads_c1_index = grads_c_index // self.c0_size
            grads_c0_index = grads_c_index % self.c0_size

            return grads_2_matrix(
                batch_indices,
                hw_mad_1_index,
                grads_c1_index,
                hw_mad_0_index,
                grads_c0_index
            )

        if self.flag_load3d_w_split_case:
            func_name = __grads_2_fractal_w_split_compute
        elif self.grads_dtype == "float32":
            func_name = __grads_2_zz_fractal_compute
        else:
            func_name = __grads_2_fractal_compute

        return tvm.compute(grads_shape_fractal,
                            lambda *indices:
                            func_name(indices, grads_2_matrix),
                            name='grads_2_fractal',
                            tag='grads_2_fractal')

    def _fmap_2_ub(self, fmap_ub_shape, fmap_height, fmap_width):
        pad_top, _, pad_left, _ = self.pad
        return tvm.compute(fmap_ub_shape,
                           lambda n, c1, h, w, c0:
                           tvm.select(tvm.any(h < pad_top,
                                              h > fmap_height + pad_top - 1,
                                              w < pad_left,
                                              w > fmap_width + pad_left - 1),
                                      tvm.const(0, self.fmap_dtype),
                                      self.fmap(n, c1, h - pad_top, w - pad_left, c0)),
                           name='fmap_ub_for_dma',
                           tag='fmap_ub_for_dma')

    def _fmap_2_matrix(self, fmap_shape_original_matrix, fmap, fmap_dtype):
        """
        compute definiton of set_fmatrix

        Parameters:
        ----------
        fmap_shape_original_matrix : shape of result tensor in L1
        in shape (batch_size,
                  grads_height*grads_width,
                  fmap_channel_1,
                  kernel_height,
                  kernel_width,
                  fmap_c0)

        fmap : input tensor in L1

        fmap_dtype : data type of fmap
        in shape (batch_size, fmap_channel_1, fmap_height, fmap_width, C0)

        Returns
        -------
        None
        """

        def __fmap_2_matrix_w_split_compute(indices, fmap):
            """
            do coordinate calculation for w-split scene.

            """
            (batch_indices,
            ho_indices,
            wo_indices,
            fmap_c1_indices,
            kernel_height_indices,
            kernel_width_indices,
            fmap_c0_indices) = indices

            n_index = batch_indices
            c1_index = fmap_c1_indices
            h_index = ho_indices * strideh + kernel_height_indices * dilationh
            w_index = wo_indices * stridew + kernel_width_indices * dilationw
            c0_index = fmap_c0_indices

            # if index belongs to padding and 16 align, select 0
            return tvm.select(tvm.any(h_index < pad_top,
                                      h_index > fmap_height + pad_top - 1,
                                      w_index < pad_left,
                                      w_index > fmap_width + pad_left - 1),
                              tvm.const(0.0, fmap_dtype),
                              fmap(n_index, c1_index, h_index - pad_top,
                                   w_index - pad_left, c0_index))

        def __fmap_2_matrix_compute(indices, fmap):
            """
            do coordinate calculation

            """
            (batch_indices,
            hw_fuse_indices,
            fmap_c1_indices,
            kernel_height_indices,
            kernel_width_indices,
            fmap_c0_indices) = indices

            n_index = batch_indices
            c1_index = fmap_c1_indices
            h_index = (hw_fuse_indices // width_out) * strideh + kernel_height_indices * dilationh
            w_index = (hw_fuse_indices % width_out) * stridew + kernel_width_indices * dilationw
            c0_index = fmap_c0_indices

            if self.l0b_dma_flag:
                return fmap(n_index, c1_index, h_index, w_index, c0_index)

            # if index belongs to padding and 16 align, select 0
            return tvm.select(tvm.any(h_index < pad_top,
                                      h_index > fmap_height + pad_top - 1,
                                      w_index < pad_left,
                                      w_index > fmap_width + pad_left - 1),
                              tvm.const(0.0, fmap_dtype),
                              fmap(n_index, c1_index, h_index - pad_top,
                                   w_index - pad_left, c0_index))

        _, _, fmap_height, fmap_width, _ = self.shapelist.get('fmap_5hd')
        pad_top, _, pad_left, pad_right = self.pad
        strideh, stridew = self.stride
        _, _, dilationh, dilationw = self.dilation
        kernel_width = fmap_shape_original_matrix[4]
        dilation_kernel_width = kernel_width + (kernel_width - 1) * (dilationw - 1)
        fmap_width_after_pad = fmap_width + pad_left + pad_right
        width_out = (fmap_width_after_pad - dilation_kernel_width) // stridew + 1

        compute_func = __fmap_2_matrix_w_split_compute if self.flag_load3d_w_split_case else __fmap_2_matrix_compute
        return tvm.compute(fmap_shape_original_matrix,
                           lambda *indices:
                           compute_func(indices, fmap),
                           name='fmap_2_col_matrix',
                           tag='fmap_2_col_matrix',
                           attrs={'pad': self.pad,
                                  'stride': self.stride,
                                  'dilation': self.dilation,
                                  'kernel_size': self.weight_shape,
                                  'group_dict': self.group_dict})

    def _fmap_2_matrix_load2d(self, fmap_shape_matrix, fmap):
        """
        compute definiton of loading fmap to L1

        Parameters:
        ----------
        fmap_shape_matrix : shape of result tensor in L1

        fmap : input tensor in ddr

        Returns
        -------
        None
        """
        def __fmap_2_matrix_load2d_compute(indices, fmap):
            """
            do coordinate calculation

            """
            fmap_width = self.shapelist.get('fmap_5hd')[3]
            batch_indices, fmap_c1_indices, hw_mad_indices, fmap_c0_indices \
                = indices
            batch_size_index = batch_indices
            fmap_c1_index = fmap_c1_indices
            fmap_height_index = (hw_mad_indices // fmap_width)
            fmap_width_index = (hw_mad_indices % fmap_width)
            fmap_c0_index = fmap_c0_indices
            return fmap(batch_size_index, fmap_c1_index,
                        fmap_height_index, fmap_width_index, fmap_c0_index)
        return tvm.compute(fmap_shape_matrix,
                           lambda *indices:
                           __fmap_2_matrix_load2d_compute(indices, fmap),
                           name='fmap_2_matrix',
                           tag='fmap_2_matrix',
                           attrs={'pad': self.pad, 'stride': self.stride,
                                  'dilation': self.dilation,
                                  'kernel_size': self.weight_shape,
                                  'group_dict': self.group_dict})

    def _fmap_2_fractal(self, fmap_shape_fmap_matrix,
                        fmap_2_col_matrix, fmap_dtype):
        """
        compute definiton of loading fmap to L0B

        Parameters:
        ----------
        fmap_shape_fmap_matrix : shape of result tensor in L0B
        in shape (batch_size,
                  hw_mad//block_size_K,
                  fmap_channel_1*kernel_height*kernel_width,
                  fmap_c0,
                  block_size_K)

        fmap_2_col_matrix : input tensor in L1
        in shape (batch_size,
                  grads_height*grads_width,
                  fmap_channel_1,
                  kernel_height,
                  kernel_width,
                  fmap_c0)

        fmap_dtype : data type of fmap_2_col_matrix


        Returns
        -------
        None
        """
        def __fmap_2_fractal_w_split_compute(indices, fmap_2_col_matrix):
            """
            do coordinate calculation for w-split scene.

            """
            _, _, grads_width, _, kernel_height, kernel_width, _ = self.shapelist.get('fmap_original_matrix')

            (group_index, n_vm_index,
            ho_indices, wo_mad_1_indices, fkk_indices,
            fmap_c0_indices, wo_mad_0_indices) = indices

            if self.fmap_dtype == "float32":
                wo_vm_index = wo_mad_1_indices * self.c0_size + wo_mad_0_indices
                c1_index = group_index * self.group_dict.get("cin1_g") + \
                    fkk_indices // (kernel_height * kernel_width) * 2 + fmap_c0_indices // self.c0_size
                kh_vm_index = fkk_indices // kernel_width % kernel_height
                kw_vm_index = fkk_indices % kernel_width
                c0_vm_index = fmap_c0_indices % self.c0_size
            else:
                wo_vm_index = wo_mad_1_indices * BLOCK_SIZE + wo_mad_0_indices
                c1_vm_index = (((fkk_indices * BLOCK_SIZE + fmap_c0_indices)
                                // BLOCK_SIZE) // kernel_width) // kernel_height
                c1_index = group_index * self.group_dict.get("cin1_g") + c1_vm_index
                kh_vm_index = (((fkk_indices * BLOCK_SIZE + fmap_c0_indices) // BLOCK_SIZE)
                                // kernel_width) % kernel_height
                kw_vm_index = ((fkk_indices * BLOCK_SIZE + fmap_c0_indices) // BLOCK_SIZE) % kernel_width
                c0_vm_index = (fkk_indices * BLOCK_SIZE + fmap_c0_indices) % BLOCK_SIZE

            # select padding and 16 align
            return tvm.select(tvm.any(wo_vm_index < 0,
                                      wo_vm_index > grads_width - 1),
                              tvm.const(0.0, fmap_dtype),
                              fmap_2_col_matrix(n_vm_index, ho_indices, wo_vm_index,
                                                c1_index, kh_vm_index,
                                                kw_vm_index, c0_vm_index))

        def __fmap_2_fractal_compute(indices, fmap_2_col_matrix):
            """
            do coordinate calculation

            """

            _, hw_fuse, _, kernel_height, kernel_width, _ \
                = self.shapelist.get('fmap_original_matrix')

            # batch_size
            # hw_mad//block_size_K
            # fmap_channel_1*kernel_height*kernel_width
            # fmap_c0
            # block_size_K
            (group_index, n_vm_index,
                hw_mad_1_indices, fkk_indices,
                fmap_c0_indices, hw_mad_0_indices) = indices

            # Dma Mode, set fractal be zZ first
            if self.l0b_dma_flag:
                (group_index, n_vm_index,
                 hw_mad_1_indices, fkk_indices,
                 hw_mad_0_indices, fmap_c0_indices) = indices

            if self.fmap_dtype == "float32":
                # matrix: fmap_c0 aligned to 8
                # fractal: hw_mad aligned to 8, while fmap_c0 aligned to 16
                # e.g., fmap is (1, 16, 7, 7) and kernel is (16, 16, 3, 3)
                # matrix is (1, 49, 2, 3, 3, 8) to fractal is (1, 1, 8, 9, 16, 8)
                hw_vm_index = hw_mad_1_indices * self.c0_size + hw_mad_0_indices
                c1_index = group_index * self.group_dict.get("cin1_g") + \
                    fkk_indices // (kernel_height * kernel_width) * 2 + fmap_c0_indices // self.c0_size
                kh_vm_index = fkk_indices // kernel_width % kernel_height
                kw_vm_index = fkk_indices % kernel_width
                c0_vm_index = fmap_c0_indices % self.c0_size
            else:
                hw_vm_index = hw_mad_1_indices*BLOCK_SIZE + hw_mad_0_indices
                c1_vm_index = (((fkk_indices*BLOCK_SIZE + fmap_c0_indices)
                                // BLOCK_SIZE) // kernel_width) // kernel_height

                c1_index = group_index * self.group_dict.get("cin1_g") + c1_vm_index

                kh_vm_index = (((fkk_indices*BLOCK_SIZE + fmap_c0_indices)
                                // BLOCK_SIZE) // kernel_width) % kernel_height
                kw_vm_index = ((fkk_indices*BLOCK_SIZE + fmap_c0_indices)
                            // BLOCK_SIZE) % kernel_width
                c0_vm_index = \
                    (fkk_indices*BLOCK_SIZE + fmap_c0_indices) % BLOCK_SIZE

            # 16 align
            if self.l0b_dma_flag:
                return tvm.select(tvm.any(hw_vm_index < hw_fuse),
                                  fmap_2_col_matrix(n_vm_index, hw_vm_index,
                                                    c1_index, kh_vm_index,
                                                    kw_vm_index, c0_vm_index))

            # select padding and 16 align
            return tvm.select(tvm.any(hw_vm_index < 0,
                                      hw_vm_index > hw_fuse - 1),
                              tvm.const(0.0, fmap_dtype),
                              fmap_2_col_matrix(n_vm_index, hw_vm_index,
                                                c1_index, kh_vm_index,
                                                kw_vm_index, c0_vm_index))

        compute_func = __fmap_2_fractal_w_split_compute if self.flag_load3d_w_split_case else __fmap_2_fractal_compute
        return tvm.compute(fmap_shape_fmap_matrix,
                           lambda *indices:
                           compute_func(indices, fmap_2_col_matrix),
                           name='fmap_2_col_fractal',
                           tag='fmap_2_col_fractal')

    def _fmap_2_fractal_load2d(self, fmap_shape_fractal, fmap_2_matrix):
        """
        compute definiton of loading fmap_matrix to L0B

        Parameters:
        ----------
        fmap_shape_fractal : shape of result tensor in L0B

        fmap_2_matrix : input tensor in L1

        Returns
        -------
        None
        """

        def __fmap_2_fractal_load2d_compute(indices, fmap_2_matrix):
            """
            do coordinate calculation

            """
            (group_indices, batch_indices,
                hw_mad_1_indices, fmap_c1_indices,
                fmap_c0_indices, hw_mad_0_indices) = indices

            batch_size_index = batch_indices
            fmap_c1_index = fmap_c1_indices
            c1_index = (
                            group_indices * self.group_dict.get("cin1_g")
                        ) + fmap_c1_index

            fmap_hw_index = (
                                hw_mad_1_indices * BLOCK_SIZE
                            ) + hw_mad_0_indices
            fmap_c0_index = fmap_c0_indices

            return fmap_2_matrix(batch_size_index, c1_index,
                                    fmap_hw_index, fmap_c0_index)

        return tvm.compute(fmap_shape_fractal,
                            lambda *indices:
                            __fmap_2_fractal_load2d_compute
                            (indices, fmap_2_matrix),
                            name='famp_2_fractal',
                            tag='famp_2_fractal')

    def _mad(self, mad_shape, grads, fmap):
        """
        calculate mad result tensor
        Parameters
        ----------
        mad_shape : result shape
        (fmap_channel_1*kernel_height*kernel_width, grads_channel, fmap_c0)

        grads : tensor in L0A
        grads_shape_fractal = (batch_size,
                               grads_channel_1,
                               hw_mad//block_size_K,
                               grads_c0,
                               block_size_K)

        fmap : tensor in L0B
        fmap_shape_fmap_matrix = (batch_size,
                                  hw_mad//block_size_K,
                                  fmap_channel_1*kernel_height*kernel_width,
                                  fmap_c0,
                                  block_size_K)

        Returns
        -------
        None
        """

        batch_size, _, grads_height, grads_width, _ = self.shapelist.get('grads_5hd')

        batch_axis = tvm.reduce_axis((0, batch_size), name='axis_b')
        reduce_axes = [batch_axis, ]
        if self.flag_load3d_w_split_case:
            k_axis_ho = tvm.reduce_axis((0, grads_height), name='axis_k_ho')
            k_axis = tvm.reduce_axis((0, grads_width), name='axis_k_wo')
            reduce_axes.extend([k_axis_ho, k_axis])
        else:
            k_axis = tvm.reduce_axis((0, grads_height * grads_width), name='axis_k')
            reduce_axes.append(k_axis)

        k_1 = k_axis.var // self.c0_size
        k_0 = k_axis.var % self.c0_size

        mode_dict = {
            ("float16", "float16"): "f162f16",
            ("float16", "float32"): "f162f32",
            ("float32", "float32"): "fp322fp32"
        }
        mode = mode_dict.get((self.fmap_dtype, self.res_dtype), "f162f32")

        if self.flag_load3d_w_split_case:
            mad_func = lambda g, fkk, grads_c, fmap_c0: tvm.sum(
                (grads[g, batch_axis, grads_c // 16, k_axis_ho, k_1, grads_c % 16, k_0] *
                fmap[g, batch_axis, k_axis_ho, k_1, fkk, fmap_c0, k_0]).astype(self.res_dtype),
                axis=reduce_axes)
        else:
            mad_func = lambda g, fkk, grads_c, fmap_c0: tvm.sum(
                (grads[g, batch_axis, grads_c // 16, k_1, grads_c % 16, k_0] *
                fmap[g, batch_axis, k_1, fkk, fmap_c0, k_0]).astype(self.res_dtype),
                axis=reduce_axes)

        # Enum for split_axis_mode
        # 0: split hw (default)
        # 1: split w
        split_axis_mode = 1 if self.flag_load3d_w_split_case else 0
        c_col = tvm.compute(mad_shape,
                            mad_func,
                            name='dw_ddr',
                            tag=self.optag + "dw_ddr",
                            attrs={
                                'mode': mode,
                                'kernel_name': self.kernel_name,
                                'split_axis_mode': split_axis_mode
                            })
        return c_col

    def _im2col_fractal_v2(self, shape, img2col_para):
        """
        compute definiton of loading fmap to L0B v2,
        only one-step compute description from l1 to l0b

        Parameters:
        ----------
        fmap_shape_fmap_matrix : shape of result tensor in L0B
        in shape (batch_size,
                  hw_mad//block_size_K,
                  fmap_channel_1*kernel_height*kernel_width,
                  fmap_c0,
                  block_size_K)

        fmap_2_col_matrix : input tensor in L1
        in shape (batch_size, fmap_channel_1, fmap_height, fmap_width, C0)

        Returns
        -------
        None
        """

        block_size = 16
        fmap, kernel_h, kernel_w, padding, stride, fmap_wo, _ = \
                                                                img2col_para

        def __im2col_idx(idx):
            group, batch, col_h, col_w, block_size_w, block_size_h = idx

            col_w = (col_w * block_size + block_size_w) // block_size
            virtual_h = col_h * block_size + block_size_h
            virtual_w = col_w * block_size + block_size_w

            back_c1 = virtual_w // block_size // kernel_w // kernel_h
            back_h = (virtual_h // fmap_wo) * stride[0] + \
                                             (col_w // kernel_w % kernel_h)
            back_w = (virtual_h % fmap_wo) * stride[1] + (col_w % kernel_w)
            indices = (group, batch, back_c1,
                        back_h - padding[0],
                        back_w - padding[2], block_size_w)
            return tvm.select(tvm.any(back_h < padding[0], \
                                    back_h > fmap.shape[2] + padding[0] - 1,
                                    back_w < padding[2], \
                                    back_w > fmap.shape[3] + padding[2] - 1),
                            tvm.const(0, fmap.dtype),
                            fmap(*indices))
        return tvm.compute(shape,
                        lambda *idx: __im2col_idx(idx),
                        name='img2col_fractal_v2',
                        tag='im2col_fractal_v2',
                        attrs={'pad': self.pad, 'stride': self.stride,
                                  'dilation': self.dilation,
                                  'kernel_size': self.weight_shape})


@tbe_utils.para_check.check_input_type(Tensor, Tensor, (list, tuple), dict)
def conv2d_backprop_filter_compute(input_x, out_backprop, filter_sizes, para_dict):
    """
    the DSL interface of conv2d backprop filter compute

    Parameters:
    ----------
    x : the featuremap data, tvm.placeholder, 5HD shape

    out_backprop : the grads data, tvm.placeholder, 5HD shape

    filter_sizes : 4-D shape, specifies the filter sizes

    para_dict:

    strides : 2-D shape, specifies in height and width dimension

    padding : 4-D shape, specifies in up/down/left/right dimension

    dilations : 4-D shape, specifies in batch/channel/height/width dimension

    groups : The number of filter's group. Default value is 1.

    res_dtype : the output data type

    Returns
    -------
    result tensor of conv2d_backprop_filter compute
    """

    strides = para_dict.get("strides", [1, 1])
    padding = para_dict.get("padding", [0, 0, 0, 0])
    dilations = para_dict.get("dilations", [1, 1, 1, 1])
    groups = para_dict.get("groups", 1)
    res_dtype = para_dict.get("res_dtype", "float32")
    kernel_name = para_dict.get("kernel_name", "conv2d_backprop_filter_cce")
    DynamicConv2dBpFilterParams.is_binary_flag = para_dict.get("is_binary_flag", False)
    DynamicConv2dBpFilterParams.correct_range_flag = para_dict.get("correct_range_flag", False)
    DynamicConv2dBpFilterParams.ori_tensors = para_dict.get("ori_tensors")
    DynamicConv2dBpFilterParams.op_type = para_dict.get("op_type", "Conv2DBackpropFilter")
    DynamicConv2dBpFilterParams.attrs = para_dict.get("attrs", {})

    deconv_dw_object = Conv2dBackpropFilter(input_x, out_backprop,
                                            filter_sizes,
                                            strides=strides,
                                            padding=padding,
                                            dilations=dilations,
                                            groups=groups,
                                            res_dtype=res_dtype,
                                            kernel_name=kernel_name)
    deconv_dw_object.deconv_dw_access()

    return deconv_dw_object.res_tensor


class DynamicConv2dBpFilterParams:
    """
    Class for parameters of dynamic conv2d_fp_filter.
    """

    var_map = {}
    correct_range_flag = False
    ori_tensors = {}
    tiling_info_dict = {}
    flag_all_one_case = False
    is_binary_flag = False
    op_type = ""
    attrs = {}
