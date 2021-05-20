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
conv2d DSL interface.
"""
from __future__ import division
import math
from topi.cce import util
from topi.cce.util import check_load3d_w_out_1_support
from tbe import tvm
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.platform import CUBE_MKN
from tbe.common.buildcfg import get_current_build_config
from tbe.common.utils.errormgr import error_manager_cube as err_man
from tbe.dsl.base.operation import get_te_var
from tbe.tvm.buffer_manager import get_buffer_manager
from tbe.tvm.dsl_source_info import source_info_decorator


# fmapH, fmapW must be in [1,4096]
FMAP_HW_MIN = 1
FMAP_W_MAX = 2**32-1
FMAP_H_MAX = 100000
DYNAMIC_FMAP_W_MAX = 4096

FMAP_W_MIN_SPLIT_W = 1
FMAP_W_MAX_SPLIT_W = 4294967295

# filterH, filterW must be in [1,255]
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255

# padH, padW must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

# stride must be in [1,63]
STRIDE_MIN = 1
STRIDE_MAX = 63

# dilate must be in [1,255]
DILATE_MIN = 1
DILATE_MAX = 255
CONV_SHAPE_DIM = 4

# In v200, small channel case: 4*filter_h*filter_w must be smaller than 65536.
HK_WK_C04_V200 = 65535
L1_FUSION_SCOPE = 1
DDR_SCOPE = 0

OP_TAG = "convolution_"
TENSOR_MAP = {}
DIM_MAP = {}
NAME_INDEX = [0]

def is_support_v200():
    """
    Check if Ascend610/Ascend615/Ascend710/Hi3796CV300CS version.
    ----------

    Returns
    -------
    True:  Ascend610/Ascend615/Ascend710/Hi3796CV300CS version
    False: Other version
    """
    soc_version = get_soc_spec("SOC_VERSION")
    if soc_version in ("Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS", "SD3403"):
        return True
    return False


def is_support_v220():
    """
    Check if Ascend920A version.

    Returns
    -------
    True: Ascend920A version.
    False: other version.
    """
    soc_version = get_soc_spec("SOC_VERSION")
    if soc_version == "Ascend920":
        return True
    return False


def check_conv_shape(shape_in, shape_w, pad_top, pad_bottom,
                     pad_left, pad_right, strideh, stridew, in_dtype, w_dtype,
                     optim_dict=None, dilateh=1, dilatew=1, dynamic_para=None, groups=1):
    """

    Parameters
    ----------
    shape_in: shape of data_in

    shape_w: shape of filter

    padh: the padding shape in H

    padw: the padding shape in weight

    strideh: the stride value in H

    stridew: the stride value in weight

    dilateh: the dilate value in H

    dilatew: the dilate value in weight

    optim_dict: optimize feature dict

    in_dtype : the feature map data type

    w_dtype : the weight data type

    Returns
    -------
    None

    """
    def _l1_buffer_size_check(max_feature_map_l1, dynamic_flag=None):
        """
        Check for not bigger than L1 size.
        """
        l1_buffer_size = get_soc_spec("L1_SIZE")
        if ConvParam.fusion_para["l1_fusion_type"] in (0, 1):
            pass
        elif int(max_feature_map_l1) > l1_buffer_size:
            if not dynamic_flag:
                ConvParam.l0a_dma_flag = True
            else:
                err_man.raise_err_specific("conv2d",
                                           "Input range is too large, the minimum tiling may exceed L1_Buffer")

    def conv1d_split_w_flag_set():
        """
        For load2d case and load3d cases, set a conv1d_split_w_flag and
        some checks do not apply to conv1D
        """
        conv1d_split_w_flag = shape_in[2] == 1 and shape_w[2] == 1 and pad_top == 0 and pad_bottom == 0
        return conv1d_split_w_flag

    def load2d_split_w_flag_set():
        """
        Set a flag for load2d to replace the load3d
        and can split w
        Some checks do not apply to conv1D.
        """
        load2d_to_load3d_flag = (shape_w[2] == 1) and (shape_w[3] == 1) and \
                                (pad_top == 0) and (pad_bottom == 0) and \
                                (pad_left == 0) and (pad_right == 0) and \
                                (strideh == 1) and (stridew == 1) and \
                                (in_dtype == "float16") and \
                                (w_dtype == "float16") and (shape_in[2] == 1)
        return load2d_to_load3d_flag

    def dilate_check():
        """
        Check dilate.
        """

        if dilateh < DILATE_MIN or dilateh > DILATE_MAX:
            range_value = "".join([str(DILATE_MIN), ", ", str(DILATE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "dilateh", str(dilateh))
        if dilatew < DILATE_MIN or dilatew > DILATE_MAX:
            range_value = "".join([str(DILATE_MIN), ", ", str(DILATE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "dilatew", str(dilatew))

    conv1d_split_w_flag = conv1d_split_w_flag_set()

    def check_fm_w_flag_set():
        """
        Check fmap split width flag.
        """
        check_fm_w_flag = False
        check_fm_w_flag = (int(shape_in[3]) < FMAP_HW_MIN or int(shape_in[3]) > FMAP_W_MAX) and not conv1d_split_w_flag
        return check_fm_w_flag

    def _check_fmap_range():
        """
        Check fmap range.
        """

        if int(shape_in[2]) < FMAP_HW_MIN:
            range_value = "".join([str(FMAP_HW_MIN), ", ", str(FMAP_H_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "feature map H", shape_in[2])
        if int(shape_in[2]) > FMAP_H_MAX:
            ConvParam.l0a_dma_flag = True
        if check_fm_w_flag_set():
            range_value = "".join([str(FMAP_HW_MIN), ", ", str(FMAP_W_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "feature map W", shape_in[3])
        if conv1d_split_w_flag and (shape_in[3] < FMAP_W_MIN_SPLIT_W or shape_in[3] > FMAP_W_MAX_SPLIT_W):
            range_value = "".join([str(FMAP_W_MIN_SPLIT_W), ", ", str(FMAP_W_MAX_SPLIT_W)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                 "feature map W when split w", shape_in[3])

    if not ConvParam.dynamic_flag:
        _check_fmap_range()
        util.check_shape_rule(shape_in, CONV_SHAPE_DIM, CONV_SHAPE_DIM)
    util.check_shape_rule(shape_w, CONV_SHAPE_DIM, CONV_SHAPE_DIM)

    if shape_in[1] != shape_w[1]:
        err_man.raise_err_scene_equal_limitation("conv2d", "input feature map channel", "filter channel")

    if optim_dict is None:
        optim_dict = {"c0_optim_flg": False, "use_v200_c04_flg": False}
    block_size_k = CUBE_MKN[in_dtype]['mac'][1]
    shape_in[1] = ((shape_in[1] + block_size_k - 1) // block_size_k)*block_size_k
    # int8 feature_map_channel_in is aligned by 16, but weight_channel_in is aligned by 32.
    shape_w[1] = ((shape_in[1] + block_size_k - 1) // block_size_k)*block_size_k
    if optim_dict["c0_optim_flg"]:
        shape_in[1] = 4
        shape_w[1] = 4
    h_i = shape_in[2]
    w_i = shape_in[3]
    h_k = shape_w[2]
    w_k = shape_w[3]

    # dilateh, dilatew check
    dilate_check()

    hk_dilation = (h_k - 1)*dilateh + 1
    wk_dilation = (w_k - 1)*dilatew + 1

    h_out = (h_i + pad_top + pad_bottom - hk_dilation) // strideh + 1
    w_out = (w_i + pad_left + pad_right - wk_dilation) // stridew + 1
    if ConvParam.dynamic_flag:
        if "fmap_h" not in dynamic_para["var_map"] and int(h_out) < 1:
            err_man.raise_err_specific("conv2d", "output shape should greater than 0, please check the input shape.\n")
        if "fmap_w" not in dynamic_para["var_map"] and int(w_out) < 1:
            err_man.raise_err_specific("conv2d", "output shape should greater than 0, please check the input shape.\n")
    else:
        if (int(w_out) < 1 or int(h_out) < 1):
            err_man.raise_err_specific("conv2d", "output shape should greater than 0, please check the input shape.\n")

    def _check_pad():
        """
        Check pad.
        """
        # padh, padw check
        if isinstance(pad_top, tvm.expr.Expr) or isinstance(pad_bottom, tvm.expr.Expr) or \
                isinstance(pad_left, tvm.expr.Expr) or isinstance(pad_right, tvm.expr.Expr):
            return
        if pad_top < PAD_MIN or pad_bottom < PAD_MIN or pad_top > PAD_MAX or pad_bottom > PAD_MAX:
            range_value = "".join([str(PAD_MIN), ", ", str(PAD_MAX)])
            actual_value = "".join([str(pad_top), ", ", str(pad_bottom)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                 "pad_top or pad_bottom", actual_value)
        if pad_left < PAD_MIN or pad_right < PAD_MIN or pad_left > PAD_MAX or pad_right > PAD_MAX:
            range_value = "".join([str(PAD_MIN), ", ", str(PAD_MAX)])
            actual_value = "".join([str(pad_left), ", ", str(pad_right)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                 "pad_left or pad_right", actual_value)

    w_block_size_n = CUBE_MKN[w_dtype]['mac'][2]
    shape_w[0] = ((shape_w[0] + w_block_size_n - 1) // w_block_size_n)*w_block_size_n

    # filterH, filterW check(before dilation according to chip design demand )
    def _check_w_range():
        """
        Check width shape.
        """
        if ConvParam.dynamic_flag:
            if shape_w[2] > FILTER_HW_MAX:
                range_value = "".join([str(FILTER_HW_MIN), ", ", str(FILTER_HW_MAX)])
                err_man.raise_err_attr_range_invalid("conv2d", range_value, "kernel H", str(shape_w[2]))
            if shape_w[3] > FILTER_HW_MAX:
                range_value = "".join([str(FILTER_HW_MIN), ", ", str(FILTER_HW_MAX)])
                err_man.raise_err_attr_range_invalid("conv2d", range_value, "kernel W", str(shape_w[3]))
        if shape_w[2] < FILTER_HW_MIN:
            range_value = "".join([str(FILTER_HW_MIN), ", ", str(FILTER_HW_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "kernel H", str(shape_w[2]))
        if shape_w[3] < FILTER_HW_MIN:
            range_value = "".join([str(FILTER_HW_MIN), ", ", str(FILTER_HW_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "kernel W", str(shape_w[3]))
        if shape_w[2] > FILTER_HW_MAX or shape_w[3] > FILTER_HW_MAX:
            ConvParam.l0a_dma_flag = True
        temp = 4*shape_w[2]*shape_w[3]
        if optim_dict.get("use_v200_c04_flg") and is_support_v200() and (temp > HK_WK_C04_V200):
            err_man.raise_err_specific("conv2d", "In v200, small channel case, the 4*Hk*Wk must be smaller than " +
                                       "or equal to " + str(HK_WK_C04_V200) +
                                       ". you can try to disable the small channel.")

    def _check_stride():
        """
        Check stride.
        """
        if strideh < STRIDE_MIN or strideh > STRIDE_MAX:
            range_value = "".join([str(STRIDE_MIN), ", ", str(STRIDE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "strideh", str(strideh))
        if stridew < STRIDE_MIN or stridew > STRIDE_MAX:
            range_value = "".join([str(STRIDE_MIN), ", ", str(STRIDE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "stridew", str(stridew))
    _check_w_range()
    _check_pad()
    _check_stride()

    config = CUBE_MKN[w_dtype]
    ci0 = config['mac'][1]
    if ci0 <= 0:
        err_man.raise_err_specific("conv2d", "ci0 must > 0")
    shape_in_fusion_para_check = shape_in
    shape_in_fusion_para_check[1] = shape_in_fusion_para_check[1]//groups

    def _check_l1_size():
        """
        check for not bigger than L1
        """

        m_bit_ratio = {"float16": 2, "int8": 1}
        if "fmap_w" in ConvParam.dyn_var_map and ConvParam.dynamic_flag:
            fmap_w_upper = get_te_var("fmap_w").get_bound()[1]
            if fmap_w_upper:
                #same as static conv2d
                wo_upper = get_te_var("wo").get_bound()[1]
                ho_upper = math.floor(config['mac'][0] / wo_upper) + 2
            else:
                #caculate ho_num for dynamic range[low, None]
                fmap_w_upper = DYNAMIC_FMAP_W_MAX
                if isinstance(pad_left, tvm.expr.Expr):
                    wo_upper = int_ceil_div(fmap_w_upper, stridew)
                else:
                    wo_upper = math.floor((fmap_w_upper - wk_dilation + pad_left + pad_right) / stridew) + 1
                ho_upper = math.floor(config['mac'][0] / wo_upper) + 2
            l1_m = ((ho_upper - 1) * strideh + hk_dilation) * fmap_w_upper
            max_feature_map_l1 = ci0 * l1_m * m_bit_ratio[w_dtype]
            _l1_buffer_size_check(max_feature_map_l1, ConvParam.dynamic_flag)
        else:
            point_per_w = math.floor((w_i - wk_dilation + pad_left + pad_right) / stridew) + 1
            w_in = math.floor(config['mac'][0] / point_per_w) + 2
            tmp = ((int(w_in) - 1) * strideh + hk_dilation) * w_i
            max_feature_map_l1 = ci0 * tmp * m_bit_ratio[w_dtype]
            if conv1d_split_w_flag:
                conv1d_filter_size = (shape_w[3] - 1) * wk_dilation + 1
                conv1d_min_l1 = (config['mac'][0] - 1) * stridew + conv1d_filter_size
                max_feature_map_l1 = ci0 * conv1d_min_l1 * m_bit_ratio[w_dtype]
            if not load2d_split_w_flag_set():
                _l1_buffer_size_check(max_feature_map_l1)
    _check_l1_size()

    return shape_in, shape_w


class ConvParam:
    """
    class of ConvParam
    """

    def __init__(self):
        pass

    def get_tensor_map(self):
        """
        Get the tensor_map in convparam.
        """
        return self.tensor_map

    @classmethod
    def set_default(cls):
        """
        Set the default value.
        """
        cls.tensor_map.clear()
        cls.dim_map.clear()
        cls.tiling = None
        cls.dyn_var_map = None
        cls.dynamic_para = None
        cls.tiling_query_param.clear()
        cls.convbn1_flag = False
        cls.conv_deq_req_double_out = False
        cls.conv_reluv2_flag = False
        cls.invalid_data_rm_flag = False
        cls.swrite_flag = False
        cls.swrite_dequant_flag = False
        cls.conv1d_split_w_flag = False
        cls.pre_relu_flag = False
        cls.strided_read_flag = False
        cls.aipp_fuse_flag = False
        cls.l0a_dma_flag = False
        cls.dynamic_flag = False
        cls.has_padding = False
        cls.dequant_doubleout_flag = False # mark v100 v200 conv_dequant_*_quant doubleout
        cls.fusion_para = {"input_memory_type": [],
                           "output_memory_type": [],
                           "l1_fusion_type": -1,
                           "fmap_l1_valid_size": -1,
                           "fmap_l1_addr_flag": "nothing",
                           "lxfusion_enable_flag": False,
                           "write_select_flag": False,
                           "allocate_root_tensor": []}

    tensor_map = {}
    dim_map = {}
    tiling = None
    tiling_query_param = {}
    convbn1_flag = False
    fusion_para = {"input_memory_type": [],
                   "output_memory_type": [],
                   "l1_fusion_type": -1,
                   "fmap_l1_valid_size": -1,
                   "fmap_l1_addr_flag": "nothing",
                   "lxfusion_enable_flag": False,
                   "write_select_flag": False,
                   "allocate_root_tensor": []}
    aipp_fuse_flag = False
    conv_reluv2_flag = False
    conv_deq_req_double_out = False
    swrite_flag = False
    strided_read_flag = False
    swrite_dequant_flag = False
    conv1d_split_w_flag = False
    pre_relu_flag = False
    l0a_dma_flag = False
    dynamic_flag = False
    has_padding = False
    dynamic_para = None
    compress_index_shape = {}
    compress_tiling_ = {}
    compress_tiling_n = {}
    compress_tiling_n_frac = {}
    compress_tiling_frac = {}
    tiling_info_dict = {}
    para_dict = {}
    dyn_var_map = {}
    fmap_range = None
    kernel_name = None


def shape_to_list(shape):
    """
    Translate tvm.shape to list type in python.
    """
    if isinstance(shape, (list, tuple)):
        return shape
    tmp = []
    for i in shape:
        if isinstance(i, tvm.expr.IntImm):
            tmp.append(int(i.value))
        elif isinstance(i, tvm.expr.Expr):
            tmp.append(i)
        else:
            tmp.append(i.value)
    return tmp


def int_ceil_div(num_a, num_b):
    """
    upper division
    """
    if num_b == 0:
        err_man.raise_err_specific("conv2d", "num_b == 0")
    return (num_a + num_b - 1) // num_b


def ceil(num_a, num_b):
    """
    upper align
    """
    if num_b == 0:
        err_man.raise_err_specific("conv2d", "division by zero")
    return (num_a + num_b - 1) // num_b*num_b


def _fmap_c0_check_value(dtype, optim_dict):
    """
    This is fmap c0 check value.
    """
    fmap_c0_check_value = 4 if optim_dict["c0_optim_flg"] and optim_dict["use_v200_c04_flg"] and is_support_v200() \
        else CUBE_MKN[dtype]['mac'][1]

    return fmap_c0_check_value

@tvm.target.generic_func
def conv_compress(inputs, weight_compress, compress_index, compress_index_shape,
                  para_dict, optim_dict=None, dsl_flag=True):
    """
    This is conv compress compute.
    """
    weight_compress_shape = weight_compress.shape
    compress_tiling_n = tvm.var("compress_tiling_n", dtype="int32")
    compress_tiling_k = tvm.var("compress_tiling_k", dtype="int32")
    compress_tiling_n_frac = tvm.var("compress_tiling_n_frac", dtype="int32")
    compress_tiling_frac = tvm.var("compress_tiling_frac", dtype="int32")
    ConvParam.compress_index_shape = compress_index_shape
    ConvParam.compress_tiling_n = compress_tiling_n
    ConvParam.compress_tiling_k = compress_tiling_k
    ConvParam.compress_tiling_n_frac = compress_tiling_n_frac
    ConvParam.compress_tiling_frac = compress_tiling_frac
    weight = tvm.compute(weight_compress_shape,
                         lambda i, j, k, l:
                         tvm.unzip(compress_index((j // compress_tiling_n * compress_tiling_n_frac + \
                                                   i // compress_tiling_k) * compress_tiling_frac * 8),
                                   weight_compress(i, j, k, l)),
                         name='weight_unzip')
    res = conv(inputs, weight, para_dict, optim_dict, dsl_flag)
    return res

@source_info_decorator()
@tvm.target.generic_func
def conv(data, weight, para_dict, optim_dict=None, dsl_flag=True):
    """
    conv

    Parameters
    ----------
    data: feature map

    weight: filter

    para_dict: dict of params

    dsl_flag: true if not from topi

    Returns
    -------
    tensor: res
    """
    def _quant_l0c2ub_compute(c_col, res_dtype):
        """
        This is l0c to ub in quant conv.
        """
        height_out = ConvParam.h_out
        width_out = ConvParam.w_out
        group = ConvParam.para_dict["group"]
        config = CUBE_MKN[in_dtype]
        block_size_m = config['mac'][0]
        howo_mad = (height_out*width_out + block_size_m - 1) // block_size_m*block_size_m

        cout1_opt = ConvParam.para_dict["cout1_opt"]
        final_c_ub_shape = (ConvParam.para_dict["a_shape"][0],
                            DIM_MAP["out_img_shape"][1], howo_mad, config['mac'][2])
        c_ub = tvm.compute(final_c_ub_shape,
                           lambda batch, cout1, howo, cout0:
                           c_col(0 if group == 1 else cout1 // cout1_opt,
                                 batch,
                                 cout1 if group == 1 else cout1 % cout1_opt,
                                 howo, cout0).astype(res_dtype),
                           name='C_UB',
                           tag=OP_TAG + "C_UB")
        if not is_support_v220():
            TENSOR_MAP["c_ub"] = c_ub
        return c_ub

    def _fmap_ddr2l1(fmap, fmap_shape, strideh_opti_flag):
        """
        This is fmap from ddr to l1.
        """
        # remove invalid fmap data in h for optimization
        if strideh_opti_flag:
            fmap_l1_shape = fmap_shape  # NC1HWC0
            fmap_l1_shape[2] = (fmap_l1_shape[2] - 1) // stride_h + 1

            if ConvParam.dynamic_flag:
                group_opt = ConvParam.para_dict["group_opt"]
                fmap_al1_shape = (group_opt, fmap_l1_shape[0], ConvParam.para_dict["c1_opt"],
                                  fmap_l1_shape[2], fmap_l1_shape[3], fmap_l1_shape[4])
                fmap_l1 = tvm.compute(fmap_al1_shape, lambda group0, n, c1, h, w, c0:
                                      fmap(n, c1 + group0*ConvParam.para_dict["c1_opt"],
                                           h*stride_h, w, c0), name="fmap_l1")
            else:
                fmap_l1 = tvm.compute(fmap_l1_shape,
                                      lambda n_idx,
                                             ci1_idx,
                                             hi_idx,
                                             wi_idx,
                                             ci0_idx:
                                      fmap[n_idx, ci1_idx, hi_idx*stride_h, wi_idx, ci0_idx],
                                      name="fmap_l1")
            TENSOR_MAP["fmap_l1"] = fmap_l1
            return fmap_l1
        return None

    def _fmap_ddr2ub(fmap, fmap_shape, padding):
        """
        This is fmap from ddr(nc1hwc0) to ub with padding(nc1hwc0)
        """
        fmap_batch, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape
        fmap_ub_shape = [fmap_batch,
                         fmap_c1,
                         fmap_h + padding[0] + padding[1],
                         fmap_w + padding[2] + padding[3],
                         fmap_c0]
        fmap_ub_for_dma_im2col = tvm.compute(fmap_ub_shape,
                                             lambda n, c1, h, w, c0:
                                             tvm.select(
                                                 tvm.any(h < padding[0],
                                                         h > fmap_h + padding[0] - 1,
                                                         w < padding[2],
                                                         w > fmap_w + padding[2] - 1),
                                                 tvm.const(offset_x, fmap.dtype),
                                                 fmap(n, c1, h - padding[0], w - padding[2], c0)),
                                             name="fmap_ub_for_dma_im2col")
        TENSOR_MAP["fmap_ub_for_dma_im2col"] = fmap_ub_for_dma_im2col
        return fmap_ub_for_dma_im2col

    def _row_major_c0_value(fmap_shape, optim_dict):
        """
        Get the c0 value in row major.
        """
        # row major res c0 value
        row_major_c0_value = 4 if optim_dict["c0_optim_flg"] else fmap_shape[4]
        return row_major_c0_value

    def _v100_cal_im2col_row_major(fmap, fmap_im2col_row_major_shape, fmap_l1, optim):
        """
        Calculate im2col row major in v100 version.
        """
        filter_w, padding, stride, dilate, strideh_opti_flag = optim

        if strideh_opti_flag:
            fmap_im2col_row_major_res = im2col_row_major(
                fmap_im2col_row_major_shape, fmap_l1,
                filter_w, padding, (1, stride_w),
                dilate, fmap.dtype)
        elif ConvParam.l0a_dma_flag and ConvParam.has_padding:
            fmap_im2col_row_major_res = im2col_row_major(
                fmap_im2col_row_major_shape,
                fmap_l1, filter_w, (0, 0, 0, 0), stride,
                dilate, fmap.dtype)
        else:
            fmap_im2col_row_major_res = im2col_row_major(
                fmap_im2col_row_major_shape,
                fmap, filter_w, padding, stride,
                dilate, fmap.dtype)
        TENSOR_MAP["fmap_im2col_row_major_res"] = fmap_im2col_row_major_res
        return fmap_im2col_row_major_res

    def _cube_compute(data, weight, mad_dtype, tiling=None, optim_dict=None, bias=None):
        """
        cube compute

        Parameters
        ----------
        data: feature map

        weight: filter

        mad_dtype: cube mad dtype

        tiling: mad tiling

        optim_dict: optim dict needed in conv

        bias: add bias after conv

        Returns
        -------
        tensor: c_col
        """
        def _config_mmad_shape():
            """
            Calculate mmad shape.
            """
            fmap_shape = shape_to_list(fmap.shape)

            batch_size = fmap_shape[0]
            feature_map_h = fmap_shape[2]
            feature_map_w = fmap_shape[3]

            height_out = ConvParam.h_out
            width_out = ConvParam.w_out

            return fmap_shape, height_out, width_out, batch_size, feature_map_h, feature_map_w

        def _mad_res(l0a_load2d_flag):
            """
            Calculate the mad result.
            """
            in_channel_c1 = ConvParam.para_dict["c1_opt"]
            in_channel_c0 = CUBE_MKN[data.dtype]['mac'][1]
            if l0a_load2d_flag:
                shape_al1_load2d = (batch_size,
                                    ConvParam.dim_map["fmap_5hd_shape"][1],
                                    feature_map_h*feature_map_w,
                                    in_channel_c0)
                al1_load2d = tvm.compute(shape_al1_load2d,
                                         lambda n, c1, m, c0:
                                         fmap(n, c1, m // feature_map_w, m % feature_map_w, c0),
                                         name=OP_TAG + "al1_load2d")
                TENSOR_MAP["al1_load2d"] = al1_load2d

                shape_al0_load2d = (
                    ConvParam.para_dict["group_opt"],
                    batch_size,
                    int_ceil_div(feature_map_h*feature_map_w,
                                 CUBE_MKN[fmap.dtype]["mac"][0]),
                    in_channel_c1,
                    CUBE_MKN[fmap.dtype]["mac"][0],
                    in_channel_c0)

                al0_load2d = tvm.compute(shape_al0_load2d,
                                         lambda group, n, m_1, c1, m_0, c0:
                                         al1_load2d(n,
                                                    group*ConvParam.para_dict["c1_opt"] + c1,
                                                    m_0 + CUBE_MKN[fmap.dtype]["mac"][0]*m_1,
                                                    c0),
                                         name=OP_TAG + "al0_load2d")
                TENSOR_MAP["al0_load2d"] = al0_load2d

                c_col = mad(mad_shape, al0_load2d, weight, config, mad_dtype)
            else:
                c_col = mad(mad_shape, fmap_im2col_fractal_res, weight, config, mad_dtype)
            return c_col

        def _get_l0a_load2d_flag():
            """
            Get the l0a_load2d_flag.
            """
            l0a_load2d_flag = False

            if (list(padding) == [0, 0, 0, 0]) and (stride == (1, 1) \
                    and w_dtype == "float16") and (filter_h*filter_w == 1):
                l0a_load2d_flag = True
                optim_dict["c0_optim_flg"] = False

            if ConvParam.fusion_para["l1_fusion_type"] in (0, 1) or ConvParam.fusion_para["input_memory_type"][0] == 1:
                l0a_load2d_flag = False

            if ConvParam.l0a_dma_flag:
                l0a_load2d_flag = False

            _, cin_ori, _, _ = ConvParam.para_dict["weight_ori_shape_nchw"]
            _, _, _, cin0 = shape_to_list(weight.shape)
            if ConvParam.para_dict.get("group_opt") > 1 and is_support_v200() and \
                (ConvParam.para_dict.get("group_opt")*ConvParam.para_dict.get("c1_opt")*cin0 != cin_ori):
                l0a_load2d_flag = False

            return l0a_load2d_flag

        def _cal_im2col_res(height_out, width_out):
            """
            Calculate im2col result
            Parameters
            ----------
            height_out: conv result height_out

            width_out: conv result width_out

            Returns
            -------
            tensor: im2col result tensor
            """
            if not ConvParam.dynamic_flag:
                in_channel_c1 = ConvParam.para_dict["c1_opt"]
                fmap_im2col_row_major_shape = (ConvParam.para_dict["group_opt"],
                                               ConvParam.para_dict["a_shape"][0],
                                               height_out*width_out,
                                               in_channel_c1,
                                               filter_h,
                                               filter_w,
                                               in_channel_c0_row_major_res)
                fmap_im2col_row_major_res = _v100_cal_im2col_row_major(fmap,
                                                                       fmap_im2col_row_major_shape,
                                                                       fmap_l1,
                                                                       [filter_w, padding, stride,
                                                                        dilate, strideh_opti_flag])

                # im2col
                # small-z-big-Z
                howo_mad = (height_out*width_out + block_size_m - 1) // block_size_m*block_size_m
                k_size = (in_channel_c0_row_major_res*in_channel_c1*filter_h*filter_w + block_size_k - 1) // \
                         block_size_k
                fmap_im2col_fractal_shape = (ConvParam.para_dict["group_opt"],
                                             ConvParam.para_dict["a_shape"][0],
                                             howo_mad // block_size_m,
                                             k_size,
                                             block_size_m,
                                             block_size_k)
                fmap_im2col_fractal_res = im2col_fractal(
                    fmap_im2col_fractal_shape, fmap_im2col_row_major_res,
                    config, fmap.dtype)

                if is_support_v200() and not c04_v100_flag and not ConvParam.l0a_dma_flag:
                    in_channel_c0 = data.shape[4].value
                    input_k_block = (in_channel_c1*filter_h*filter_w*in_channel_c0 + block_size_k - 1) // \
                                    block_size_k*block_size_k
                    row_major_reshape_shape = (ConvParam.para_dict["group_opt"], batch_size, howo_mad, input_k_block)
                    row_major_reshape_res = _im2col_row_major_reshape(row_major_reshape_shape,
                                                                      fmap_im2col_row_major_res,
                                                                      fmap.dtype)
                    fmap_im2col_fractal_res = _im2col_fractal_v200(fmap_im2col_fractal_shape,
                                                                   row_major_reshape_res,
                                                                   config)
                    TENSOR_MAP["row_major_reshape_res"] = row_major_reshape_res
                TENSOR_MAP["fmap_im2col_fractal_res"] = fmap_im2col_fractal_res
            else:
                in_channel_c1 = ConvParam.para_dict["c1_opt"]
                howo_mad = (height_out * width_out + block_size_m - 1) // block_size_m * block_size_m
                fmap_im2col_fractal_shape = (ConvParam.para_dict["group_opt"],
                                             ConvParam.para_dict["a_shape"][0],
                                             howo_mad // block_size_m,
                                             in_channel_c1 * filter_h * filter_w,
                                             block_size_m,
                                             block_size_k)
                if not strideh_opti_flag:
                    if ConvParam.dynamic_flag:
                        group_opt = ConvParam.para_dict["group_opt"]
                        fmap_shape = ConvParam.para_dict["a_shape"]
                        fmap_l1_shape = (group_opt, fmap_shape[0], ConvParam.para_dict["c1_opt"],
                                         fmap_shape[2], fmap_shape[3], fmap_shape[4])
                        fmap_al1 = tvm.compute(fmap_l1_shape, lambda group0, n, c1, h, w, c0:
                                               fmap(n, c1 + group0*ConvParam.para_dict["c1_opt"],
                                                    h, w, c0), name="fmap_l1")
                        img2col_para = (fmap_al1, filter_h, filter_w, padding, stride, width_out)
                        TENSOR_MAP["fmap_l1"] = fmap_al1
                    else:
                        img2col_para = (fmap, filter_h, filter_w, padding, stride, width_out)
                else:
                    img2col_para = (fmap_l1, filter_h, filter_w, padding, (1, stride_w), width_out)
                fmap_im2col_fractal_res = img2col(fmap_im2col_fractal_shape, img2col_para)
                TENSOR_MAP["fmap_im2col_fractal_res"] = fmap_im2col_fractal_res

            return howo_mad, fmap_im2col_fractal_res

        def _cal_bias_res():
            """
            Calculate bias result.
            """
            config = CUBE_MKN[w_dtype]
            # load bias into UB and do 32Byte align
            bias_32byte_align_shape = []
            bias_32byte_align_shape.append(ceil(bias_tensor.shape[0], 8))
            bias_ub = tvm.compute(bias_32byte_align_shape, lambda *indice: bias_tensor(*indice), name='bias_ub')
            if bias_optimize_flag:
                bias_ub_brc_shape = list(mad_shape)
                bias_ub_brc_shape[3] = bias_ub_brc_shape[3] // 16
                bias_ub_brc = tvm.compute(
                    bias_ub_brc_shape,
                    lambda group, i, j, k, l:
                    bias_ub(group * bias_ub_brc_shape[2] * config['mac'][2] + j * config['mac'][2] + l),
                    name=OP_TAG + 'bias_ub_brc')
                bias_l0c = tvm.compute(
                    mad_shape,
                    lambda group, i1, j1, k_1, l1:
                    bias_ub_brc(group, i1, j1, k_1 // 16, l1),
                    name=OP_TAG + 'bias_l0c')
                TENSOR_MAP["bias_ub_brc"] = bias_ub_brc
                TENSOR_MAP["bias_l0c"] = bias_l0c
            else:
                bias_l0c = tvm.compute(
                    mad_shape,
                    lambda group, i1, j1, k_1, l1:
                    bias_ub(group * mad_shape[2] * config['mac'][2] + j1 * config['mac'][2] + l1),
                    name=OP_TAG + 'bias_l0c')
                TENSOR_MAP["bias_l0c"] = bias_l0c

            TENSOR_MAP["bias_optimize_flag"] = bias_optimize_flag

            c_col = tvm.compute(
                mad_shape,
                lambda *index:
                bias_l0c(*index) + TENSOR_MAP["c_col"](*index),
                name=OP_TAG + 'c_col_bias')
            TENSOR_MAP["c_col_bias"] = c_col
            TENSOR_MAP["bias_ub"] = bias_ub
            return bias_l0c, c_col

        fmap = data
        in_dtype = fmap.dtype

        weight_shape = shape_to_list(weight.shape)
        out_channel_c1 = weight_shape[1]
        out_channel_c0 = weight_shape[2]
        out_channel = out_channel_c1*out_channel_c0

        if ConvParam.dynamic_flag and ConvParam.para_dict["pooling_mode"] == "AVG":
            strideh_opti_flag = (filter_h == 1 and stride_h > 1) and not optim_dict["c0_optim_flg"]
        else:
            strideh_opti_flag = (filter_h == 1 and stride_h > 1) and not optim_dict["c0_optim_flg"] and \
                sum(ConvParam.para_dict['pad_h'] + ConvParam.para_dict['pad_w']) == 0

        if ConvParam.fusion_para["l1_fusion_type"] == 1 or ConvParam.fusion_para["input_memory_type"][0] == 1:
            # for L1 breadth fusion, fmap must load all at once
            strideh_opti_flag = False

        if ConvParam.pre_relu_flag or ConvParam.l0a_dma_flag:
            strideh_opti_flag = False

        padding = ConvParam.padding
        stride = (stride_h, stride_w)
        c04_v100_flag = optim_dict["c0_optim_flg"] and not (is_support_v200() and optim_dict["use_v200_c04_flg"])
        c04_v200_flag = optim_dict["c0_optim_flg"] and is_support_v200() and optim_dict["use_v200_c04_flg"]
        TENSOR_MAP["strideh_opti_flag"] = strideh_opti_flag
        TENSOR_MAP["c0_optim_flg"] = c04_v100_flag
        TENSOR_MAP["c04_v200_flag"] = c04_v200_flag

        l0a_load2d_flag = _get_l0a_load2d_flag()
        TENSOR_MAP["l0a_load2d_flag"] = l0a_load2d_flag

        fmap_shape, height_out, width_out, batch_size, feature_map_h, feature_map_w = _config_mmad_shape()
        config = CUBE_MKN[in_dtype]
        block_size_k = config['mac'][1]
        block_size_m = config['mac'][0]
        dilate = (dilate_h, dilate_w)

        if ConvParam.l0a_dma_flag and ConvParam.has_padding:
            # DDR -> UB
            fmap_l1 = _fmap_ddr2ub(fmap, fmap_shape, padding)
        else:
            # DDR -> L1
            fmap_l1 = _fmap_ddr2l1(fmap, fmap_shape, strideh_opti_flag)

        # set_fmatrix
        # calculate im2col_row_major
        in_channel_c0_row_major_res = _row_major_c0_value(fmap_shape, optim_dict)
        howo_mad, fmap_im2col_fractal_res = _cal_im2col_res(height_out, width_out)

        config = CUBE_MKN[res_dtype]

        mad_shape = (ConvParam.para_dict["group_opt"],
                     ConvParam.para_dict["a_shape"][0],
                     ConvParam.para_dict["cout1_opt"], howo_mad, config['mac'][2])

        config = CUBE_MKN[w_dtype]
        ConvParam.mad_shape = mad_shape

        # set height_width value for mad to use.
        DIM_MAP["out_img_height_width"] = [height_out, width_out]

        c_col = _mad_res(l0a_load2d_flag)

        TENSOR_MAP["c_col"] = c_col

        DIM_MAP["out_img_shape"] = (
            ConvParam.para_dict["a_shape"][0],
            int_ceil_div(ConvParam.para_dict["weight_ori_shape_nchw"][0], 16),
            ConvParam.h_out*ConvParam.w_out,
            16)

        if ConvParam.v200_width_out_1_flag: # in special case, actual out size is only half
            DIM_MAP["out_img_shape"] = (
                ConvParam.para_dict["a_shape"][0],
                int_ceil_div(ConvParam.para_dict["weight_ori_shape_nchw"][0], 16),
                ConvParam.h_out*ConvParam.w_out // 2,
                16)
        ConvParam.conv_shape = DIM_MAP["out_img_shape"]
        filter_shape = [out_channel, filter_h, filter_w, 1]
        dim_map1 = im2col_dim(shape_to_list(fmap.shape),
                              filter_shape, list(padding), list(stride),
                              list(dilate), config)
        dim_map_copy = DIM_MAP.copy()
        dim_map_copy.update(dim_map1)
        ConvParam.dim_map.update(dim_map_copy)
        ConvParam.tiling = tiling
        TENSOR_MAP["conv_vector_fused_flag"] = False
        TENSOR_MAP["bias_optimize_flag"] = False

        bias_tensor_flag = isinstance(bias, tvm.tensor.Tensor)
        bias_optimize_flag = True
        if is_support_v200():
            bias_optimize_flag = False
        if ConvParam.dynamic_flag:
            bias_optimize_flag = False

        howo_mad = (height_out*width_out + block_size_m - 1) // block_size_m*block_size_m

        mad_shape = (ConvParam.para_dict["group_opt"],
                     ConvParam.para_dict["a_shape"][0],
                     ConvParam.para_dict["cout1_opt"], howo_mad, config['mac'][2])

        if bias_tensor_flag:
            _, c_col = _cal_bias_res()

        ConvParam.tensor_map = TENSOR_MAP
        return c_col

    def cub_fp16_compute(data, weight, mad_dtype, res_dtype, stride_h,
                         stride_w, dilate_h, dilate_w, filter_h, filter_w, bias=False,
                         no_vector=False, tiling=None, conv_fused_flag=False,
                         optim_dict=None, kernel_name=None, padding_mode="VALID", pooling_mode=None):
        """
        conv

        Parameters
        ----------
        data: tvm.tensor, Feature Map

        weight: tvm.tensor, Filter

        res_dtype : the result data type

        pad_h: the padding shape in height

        pad_w: the padding shape in weight

        stride_h: the stride value in height

        stride_w: the stride value in weight

        dilate_h: the dilate value in H

        dilate_w: the dilate value in Weight

        filter_h: kernel size of height

        filter_w: kernel_size of weight

        bias: the tag for bias or not

        drq_scale: scale tensor of DeQuant or ReQuant

        scalr_vector_flag: the tag for scalar mode or vector mode

        offset_pad: offset_pad tensor for ReQuant in half-offset mode

        no_vector: the tag for conv has vector compute or not

        tiling: default none, tiling

        conv_fused_flag: the tag indicates conv fusion
        -------
        Returns

        wrapped_tensor
        """
        fmap = data
        in_dtype = fmap.dtype

        weight_shape = shape_to_list(weight.shape)
        out_channel_c1 = weight_shape[1]
        out_channel_c0 = weight_shape[2]
        out_channel = out_channel_c1*out_channel_c0

        height_out = ConvParam.h_out
        width_out = ConvParam.w_out

        config = CUBE_MKN[in_dtype]
        block_size_m = config['mac'][0]
        padding = ConvParam.padding
        stride = (stride_h, stride_w)
        dilate = (dilate_h, dilate_w)

        c_col = _cube_compute(fmap, weight, mad_dtype,
                              tiling, optim_dict, bias)

        howo_mad = (height_out*width_out + block_size_m - 1) // block_size_m*block_size_m
        cout1_opt = ConvParam.para_dict["cout1_opt"]
        final_c_ub_shape = (ConvParam.para_dict["a_shape"][0],
                            DIM_MAP["out_img_shape"][1], howo_mad, config['mac'][2])
        config = CUBE_MKN[w_dtype]

        group = ConvParam.para_dict["group"]

        c_ub = tvm.compute(final_c_ub_shape,
                           lambda batch, cout1, howo, cout0:
                           c_col(0 if group == 1 else cout1 // cout1_opt,
                                 batch,
                                 cout1 if group == 1 else cout1 % cout1_opt,
                                 howo, cout0).astype(res_dtype),
                           name='C_UB',
                           tag=OP_TAG + "C_UB",
                           attrs={'no_vector': no_vector,
                                  'sqrt': False,
                                  'res_dtype': res_dtype,
                                  'kernel_h': filter_h,
                                  'kernel_w': filter_w,
                                  'padding': padding,
                                  'stride': stride,
                                  'dilate': dilate,
                                  'width_out': width_out,
                                  'kernel_name': kernel_name})
        if pooling_mode == "AVG":
            out_h = ConvParam.h_out
            out_w = ConvParam.w_out
            pad_t, _, pad_l, _ = padding
            input_h, input_w = data.shape[2:4]
            conv_shape = ConvParam.dim_map["output_conv_res_shape"]
            if padding_mode == "VALID":
                c_ub_avg = tvm.compute(conv_shape,
                                       lambda n, c1, m, c0:
                                       tvm.div(c_ub(n, c1, m, c0), filter_h*filter_w).astype(res_dtype),
                                       name='C_UB_AVG',
                                       tag=OP_TAG + "C_UB_AVG")
            else:
                mean_matrix_shape = [c_ub.shape[2], c_ub.shape[3]]
                mean_matrix = tvm.compute(mean_matrix_shape,
                                          lambda m, c0:
                                          tvm.select(
                                              tvm.any(m < out_h*out_w),
                                              tvm.max(
                                                  (tvm.min((m // out_w)*stride_h-pad_t+filter_h, input_h) -
                                                   tvm.max((m // out_w)*stride_h-pad_t, 0))* \
                                                   (tvm.min((m % out_w)*stride_w-pad_l+filter_w, input_w) -
                                                    tvm.max((m % out_w)*stride_w-pad_l, 0)), 1
                                              ).astype("int")),
                                          name="mean_matrix")
                mean_matrix_fp16 = tvm.compute(mean_matrix_shape,
                                               lambda *index:
                                               mean_matrix(*index).astype(res_dtype),
                                               name="mean_matrix_fp16")
                if "Ascend310" in get_soc_spec("SOC_VERSION"):
                    mean_matrix_rec = tvm.compute(mean_matrix_shape,
                                                  lambda *index:
                                                  1/mean_matrix_fp16(*index),
                                                  name="mean_matrix_rec")
                    c_ub_avg = tvm.compute(conv_shape,
                                           lambda n, c1, m, c0:
                                           c_ub(n, c1, m, c0)*mean_matrix_rec(m, c0),
                                           name='C_UB_AVG',
                                           tag=OP_TAG + "C_UB_AVG")
                else:
                    c_ub_avg = tvm.compute(conv_shape,
                                           lambda n, c1, m, c0:
                                           tvm.div(c_ub(n, c1, m, c0), mean_matrix_fp16(m, c0)).astype(res_dtype),
                                           name='C_UB_AVG',
                                           tag=OP_TAG+"C_UB_AVG")
                    mean_matrix_rec = c_ub_avg
                TENSOR_MAP["mean_matrix_fp16"] = mean_matrix_fp16
                TENSOR_MAP["mean_matrix_rec"] = mean_matrix_rec
                TENSOR_MAP["mean_matrix"] = mean_matrix
            TENSOR_MAP["c_ub_avg"] = c_ub_avg

        filter_shape = [out_channel, filter_h, filter_w, 1]
        dim_map1 = im2col_dim(shape_to_list(fmap.shape), filter_shape,
                              list(padding), list(stride),
                              list(dilate), config)
        dim_map_copy = DIM_MAP.copy()
        dim_map_copy.update(dim_map1)

        if not is_support_v220():
            TENSOR_MAP["c_ub"] = c_ub
        TENSOR_MAP["conv_vector_fused_flag"] = conv_fused_flag
        ConvParam.tensor_map = TENSOR_MAP
        ConvParam.dim_map.update(dim_map_copy)
        ConvParam.tiling = tiling
        if pooling_mode == "AVG":
            return c_ub_avg
        return c_ub

    def img2col(shape, img2col_para):
        """
        Calculate im2col result
        Parameters
        ----------
        shape: fraz fmap shape

        img2col_para: img2col paras

        Returns
        -------
        tensor: im2col result tensor
        """
        block_size = 16
        fmap, kernel_h, kernel_w, padding, stride, fmap_wo = img2col_para

        def __im2col_idx(idx):
            """
            Calculate im2col result main compute
            Parameters
            ----------
            idx: index in fraz fmap

            Returns
            -------
            tensor: im2col result tensor
            """
            group, n_batch, col_h, col_w, block_size_h, block_size_w = idx

            virtual_h = col_h * block_size + block_size_h
            virtual_w = col_w * block_size + block_size_w

            back_c1 = virtual_w // block_size // kernel_w // kernel_h
            back_h = (virtual_h // fmap_wo) * stride[0] + (col_w // kernel_w % kernel_h)
            back_w = (virtual_h % fmap_wo) * stride[1] + (col_w % kernel_w)

            if len(fmap.shape) == len(ConvParam.para_dict["a_shape"]):
                return tvm.select(tvm.any(back_h < padding[0],
                                          back_h > fmap.shape[2] + padding[0] - 1,
                                          back_w < padding[2],
                                          back_w > fmap.shape[3] + padding[2] - 1),
                                  tvm.const(0, fmap.dtype),
                                  fmap(n_batch,
                                       back_c1 + group*ConvParam.para_dict["c1_opt"],
                                       back_h - padding[0],
                                       back_w - padding[2],
                                       block_size_w))
            return tvm.select(tvm.any(back_h < padding[0],
                                      back_h > fmap.shape[3] + padding[0] - 1,
                                      back_w < padding[2],
                                      back_w > fmap.shape[4] + padding[2] - 1),
                              tvm.const(0, fmap.dtype),
                              fmap(group,
                                   n_batch,
                                   back_c1,
                                   back_h - padding[0],
                                   back_w - padding[2],
                                   block_size_w))
        return tvm.compute(shape,
                           lambda *idx: __im2col_idx(idx),
                           name='img2col_fractal_v2',
                           tag=OP_TAG + 'im2col_fractal_v2',
                           attrs={
                               'fmap_shape': fmap.shape,
                               'kernel_h': kernel_h,
                               'kernel_w': kernel_w,
                               'padding': padding,
                               'stride': stride})

    def im2col_dim(img_shape, filter_shape, pad, stride, dilate, config):
        """
        calculate shape
        Parameters

        ----------
        img_shape : shape of feature

        filter_shape : shape of filter

        pad: the padding shape

        stride: the stride value

        dilate: the dilate value

        Returns
        -------
        img_shape, fmap_matrix_dim
        """
        mac_dim = config['mac']

        batch = img_shape[0]
        if "fmap_h" not in ConvParam.dyn_var_map:
            out_h = ((img_shape[-3] + pad[2] + pad[3]) -
                     ((filter_shape[-3]-1)*dilate[0] + 1)) // stride[0] + 1
        else:
            out_h = ConvParam.dyn_var_map['ho']
        if "fmap_w" not in ConvParam.dyn_var_map:
            out_w = ((img_shape[-2] + pad[0] + pad[1]) -
                     ((filter_shape[-2]-1)*dilate[1] + 1)) // stride[1] + 1
        else:
            out_w = ConvParam.dyn_var_map['wo']

        fmap_valid_dim = (batch,
                          out_h*out_w,
                          ConvParam.para_dict["c1_opt"]*img_shape[-1]*filter_shape[-2]*filter_shape[-3])

        fmap_matrix_dim = (batch,
                           ((fmap_valid_dim[-2] + mac_dim[0] - 1) // mac_dim[0]),
                           ((fmap_valid_dim[-1] + mac_dim[1] - 1) // mac_dim[1]),
                           mac_dim[0],
                           mac_dim[1])

        filter_valid_dim = (ConvParam.para_dict["c1_opt"]*filter_shape[-3]*filter_shape[-2]
                            * img_shape[-1], filter_shape[-4]*filter_shape[-1])

        filter_matrix_dim = ((filter_valid_dim[-2] + mac_dim[1] - 1) // mac_dim[1],
                             (filter_valid_dim[-1] + mac_dim[2] - 1) // mac_dim[2],
                             mac_dim[2],
                             mac_dim[1])

        img_shape_single_group = img_shape
        img_shape_single_group[1] = ConvParam.para_dict["c1_opt"]
        return {"img_shape": img_shape_single_group,
                "fmap_matrix_dim": fmap_matrix_dim,
                "filter_matrix_dim": filter_matrix_dim}

    def im2col_row_major(fmap_im2col_vm_shape, fmap, kernel_w, padding, stride, dilate, compute_dtype):
        """
        calculate im2col_row_major tensor

        Parameters
        ----------
        fmap_im2col_vm_shape: shape of fmap_im2col_row_major

        fmap: feature map

        kernel_w: the kernel value in w

        padding: the padding shape

        stride: the stride value

        dilate: the dilate value

        compute_dtype: dtype of compute result

        Returns
        -------
        fmap_im2col_row_major tensor
        """
        def __im2col_row_major_indices(group, batch, howo, cin_1, k_h, k_w, cin_0,
                                       fmap, kernel_w, padding, stride, dilate):

            """
            calculate im2col_row_major tvm lambda function
            Parameters
            ----------
            indices : indices in lambda function

            fmap: feature map

            padding: the padding shape

            stride: the stride value

            dilate: the dilate value

            Returns
            -------
            im2col_row_major tvm lambda function
            """
            _, _, input_h, input_w, _ = fmap.shape
            stride_h, stride_w = stride
            dilate_h, dilate_w = dilate
            padding_top, _, padding_left, padding_right = padding
            width_out = (input_w.value + padding_left + padding_right - ((kernel_w - 1)*dilate_w + 1)) // (stride_w) + 1

            h_index = (howo // width_out)*stride_h + k_h*dilate_h
            w_index = (howo % width_out)*stride_w + k_w*dilate_w
            if ConvParam.l0a_dma_flag:
                return fmap(batch,
                            cin_1 + group * ConvParam.para_dict["c1_opt"],
                            h_index - padding_top,
                            w_index - padding_left, cin_0)

            return tvm.select(
                tvm.any(h_index < padding_top,
                        h_index > input_h.value + padding_top - 1,
                        w_index < padding_left,
                        w_index > input_w.value + padding_left - 1),
                tvm.const(offset_x, compute_dtype),
                fmap(batch,
                     cin_1 + group * ConvParam.para_dict["c1_opt"],
                     h_index - padding_top,
                     w_index - padding_left, cin_0))

        return tvm.compute(fmap_im2col_vm_shape,
                           lambda group, batch, howo, cin_1, k_h, k_w, cin_0:
                           __im2col_row_major_indices(group, batch, howo, cin_1, k_h, k_w, cin_0,
                                                      fmap, kernel_w, padding, stride, dilate),
                           name='im2col_row_major',
                           tag=OP_TAG + 'im2col_row_major')

    def im2col_fractal(fmap_im2col_shape, fmap, config, compute_dtype):
        """
        calculate im2col_fractal tensor
        Parameters
        ----------
        fmap_im2col_shape : shape of fmap_im2col

        fmap : feature map

        config: the config of cube

        compute_dtype: dtype of compute result

        Returns
        -------
        fmap_im2col_fractal tensor
        """
        def __im2col_fractal_indices(group, batch, m_1, k_1, m_0, k_0, fmap):
            """
            calculate im2col_fractal tvm lambda function
            Parameters
            ----------
            indices : indices in lambda function

            fmap : feature map

            Returns
            -------
            im2col_fractal tvm lambda function
            """
            block_size = config['mac'][1]
            block_size_m = config['mac'][0]
            _, _, howo, _, kernel_h, kernel_w, _ = fmap.shape

            hw_index = m_1*block_size_m + m_0

            c1_index = (((k_1*block_size + k_0) // block_size) // kernel_w.value) // kernel_h.value

            kh_index = (((k_1*block_size + k_0) // block_size) // kernel_w.value) % kernel_h.value

            kw_index = ((k_1*block_size + k_0) // block_size) % kernel_w.value

            c0_index = (k_1*block_size + k_0) % block_size

            if optim_dict["c0_optim_flg"]:
                c1_index = 0
                kh_index = (k_1*4 + k_0 // 4) // kernel_w.value
                kw_index = (k_1*4 + k_0 // 4) % kernel_w.value
                c0_index = k_0 % 4
            dtype = compute_dtype
            if ConvParam.l0a_dma_flag:
                return fmap(group, batch, hw_index, c1_index, kh_index, kw_index, c0_index)

            return tvm.select(
                tvm.any(hw_index < 0, hw_index > howo.value - 1),
                tvm.const(0.0, dtype),
                fmap(group, batch, hw_index,
                     c1_index, kh_index, kw_index, c0_index))

        return tvm.compute(fmap_im2col_shape,
                           lambda group, batch, m_1, k_1, m_0, k_0:
                           __im2col_fractal_indices(group, batch, m_1, k_1, m_0, k_0, fmap),
                           name='im2col_fractal',
                           tag=OP_TAG + 'im2col_fractal')

    def _im2col_row_major_reshape(fmap_im2col_shape, fmap_row_major, compute_dtype):
        """
        merage im2col_row_major axis of input_C1, filter_h, filter_w, input_C0
        Parameters
        ----------
        fmap_im2col_shape : shape of fmap_im2col

        fmap_row_major : feature map after im2col_row_major

        compute_dtype: dtype of compute result

        Returns
        -------
        row_major_reshape tensor
        """
        fmap_c1 = ConvParam.para_dict["a_shape"][1]
        fmap_c0 = ConvParam.para_dict["a_shape"][4]
        kernel_h = ConvParam.para_dict["filter_h"]
        kernel_w = ConvParam.para_dict["filter_w"]
        reduce_c1hwc0 = fmap_c1*fmap_c0*kernel_h*kernel_w
        _, _, howo, input_c1, filter_h, filter_w, input_c0 = fmap_row_major.shape
        if ConvParam.para_dict["group_opt"] > 1:
            row_major_reshape = tvm.compute(
                fmap_im2col_shape,
                lambda group, i, j, k:
                tvm.select(tvm.all(k < input_c1*filter_h*filter_w*input_c0, j < howo,
                                   group*fmap_im2col_shape[3] + k < reduce_c1hwc0),
                           fmap_row_major(group, i, j, k // (filter_h*filter_w*input_c0),
                                          k // (filter_w*input_c0) % filter_h,
                                          k // (input_c0) % (filter_w),
                                          k % input_c0), tvm.const(0.0, compute_dtype)),
                name="row_major_reshape",
                tag=OP_TAG + 'row_major_reshape')
        else:
            row_major_reshape = tvm.compute(
                fmap_im2col_shape,
                lambda group, i, j, k:
                tvm.select(tvm.all(k < input_c1*filter_h*filter_w*input_c0, j < howo),
                           fmap_row_major(group, i, j, k // (filter_h*filter_w*input_c0),
                                          k // (filter_w*input_c0) % filter_h,
                                          k // (input_c0) % (filter_w),
                                          k % input_c0), tvm.const(0.0, compute_dtype)),
                name="row_major_reshape",
                tag=OP_TAG + 'row_major_reshape')

        return row_major_reshape

    def _im2col_fractal_v200(fmap_im2col_shape, im2col_row_major_reshape, config):
        """
        calculate im2col_fractal tensor
        Parameters
        ----------
        fmap_im2col_shape : shape of fmap_im2col

        im2col_row_major_reshape : feature map of _im2col_row_major_reshape

        config: the config of cube

        compute_dtype: dtype of compute result

        Returns
        -------
        fmap_im2col_fractal tensor
        """
        block_size_m = config['mac'][0]
        block_size_k = config['mac'][1]
        res_im2col_fractal = tvm.compute(fmap_im2col_shape,
                                         lambda group, i, j, k, l, m:
                                         im2col_row_major_reshape(group, i, j*block_size_m + l, k*block_size_k + m),
                                         name="_im2col_fractal",
                                         tag=OP_TAG + '_im2col_fractal')

        return res_im2col_fractal

    def mad(mad_shape, fmap, weight, config, mad_dtype):
        """
        calculate mad result tensor
        Parameters
        ----------
        mad_shape : shape of mad result

        fmap : feature map

        weight : filter

        config: the config of cube

        mad_dtype: dtype of mad output

        Returns
        -------
        mad result tensor
        """
        block_size = config['mac'][1]
        block_size_m = config['mac'][0]

        reduce_k1 = weight.shape[0] // ConvParam.para_dict["group_opt"]

        axis_k1 = tvm.reduce_axis((0, reduce_k1), name='cin_1_kh_kw')
        axis_k0 = tvm.reduce_axis((0, block_size), name='cin_0')

        if mad_dtype in ["float16", "int32"]:
            mode = 'f162f16'
        else:
            mode = 'f162f32'
        fmap_c1 = ConvParam.para_dict["a_shape"][1]
        kernel_h = ConvParam.para_dict["filter_h"]
        kernel_w = ConvParam.para_dict["filter_w"]
        reduce_c1hwc0 = fmap_c1*kernel_h*kernel_w*block_size
        if TENSOR_MAP["c0_optim_flg"]:
            reduce_c1hwc0 = fmap_c1*kernel_h*kernel_w*4
        if not ConvParam.v200_width_out_1_flag:
            remove_pad_m = DIM_MAP["out_img_height_width"][0]*DIM_MAP["out_img_height_width"][1]
        else: # invliad_data_rm uses removed_pad_m as the shape of UB tensor, so modify it to N*1 here
            remove_pad_m = DIM_MAP["out_img_height_width"][0]*DIM_MAP["out_img_height_width"][1] // 2

        offset_d = offset_x if is_support_v200() else 0
        if TENSOR_MAP["c0_optim_flg"] or \
            (ConvParam.para_dict["group_opt"] > 1 and is_support_v200() and not TENSOR_MAP["l0a_load2d_flag"]):
            c_col = tvm.compute(
                mad_shape,
                lambda group, batch, cout_1, howo, cout_0:
                tvm.sum(
                    tvm.select(tvm.all((group * reduce_k1 + axis_k1) * block_size + axis_k0 < reduce_c1hwc0),
                               ((fmap[group,
                                      batch,
                                      howo // block_size_m,
                                      axis_k1,
                                      howo % block_size_m,
                                      axis_k0] - offset_d) *
                                weight[group*reduce_k1+axis_k1,
                                       cout_1,
                                       cout_0,
                                       axis_k0]).astype(mad_dtype)),
                    axis=[axis_k1, axis_k0]),
                name='mad1',
                tag=OP_TAG + "c_col",
                attrs={'mode': mode,
                       'remove_pad_M': remove_pad_m}) # used in Feature: invalid_data_rm
        else:
            c_col = tvm.compute(
                mad_shape,
                lambda group, batch, cout_1, howo, cout_0:
                tvm.sum((
                    (fmap[group,
                          batch,
                          howo // block_size_m,
                          axis_k1,
                          howo % block_size_m,
                          axis_k0] - offset_d) *
                    weight[group*reduce_k1+axis_k1,
                           cout_1,
                           cout_0,
                           axis_k0]).astype(mad_dtype),
                        axis=[axis_k1, axis_k0]),
                name='mad1',
                tag=OP_TAG + "c_col",
                attrs={'mode': mode,
                       'remove_pad_M': remove_pad_m}) # used in Feature: invalid_data_rm

        return c_col

    def bias_add(in_tensor0, in_tensor1):
        """
        calculate conv res + bias in UB
        Parameters
        ----------
        in_tensor0: conv res tensor

        in_tensor1: bias vector

        Returns
        -------
        in_tensor0+in_tensor1 tensor
        """
        out_shape = shape_to_list(in_tensor0.shape)
        NAME_INDEX[0] += 1

        # load bias into UB and do 32Byte align
        bias_32byte_align_shape = []
        bias_32byte_align_shape.append(ceil(in_tensor1.shape[0], 16))
        bias_ub = tvm.compute(bias_32byte_align_shape, lambda *indice: in_tensor1(*indice), name='bias_ub')
        TENSOR_MAP["bias_ub"] = bias_ub

        with tvm.tag_scope('conv_vector_bias_add'):
            c_add_vector = tvm.compute(
                out_shape,
                lambda *indice:
                in_tensor0(*indice) + bias_ub(indice[1]*CUBE_MKN[in_tensor0.dtype]['mac'][2] + indice[3]),
                name='bias_add_vector' + "_cc_" + str(NAME_INDEX[0]),
                attrs={'width_out': in_tensor0.op.attrs["width_out"]})
        return c_add_vector

    def remove_pad(res, conv_shape):
        """
        remove pad
        Parameters
        ----------
        res: input tensor

        res_remove_pad_shape: true shape

        Returns
        -------
        res_remove_pad tensor
        """
        NAME_INDEX[0] += 1
        with tvm.tag_scope('conv_vector_remove_pad'):
            res_tensor = tvm.compute(conv_shape,
                                     lambda batch, cout1, howo, cout0:
                                     res(batch, cout1, howo, cout0),
                                     name='remove_pad' + "_cc_" + str(NAME_INDEX[0]))
        return res_tensor

    def remove_pad_quant_dsl(res, conv_shape, invalid_data_rm_flag, params_dict=None):
        """
        remove pad
        Parameters
        ----------
        res: input tensor.

        conv_shape: true shape.

        invalid_data_rm_flag: True for conv2d without removing pad.

        params_dict: a dict of relevant infos

        Returns
        -------
        res_remove_pad tensor
        """
        NAME_INDEX[0] += 1
        group = ConvParam.para_dict["group"]
        cout1_opt = ConvParam.para_dict["cout1_opt"]
        conv_shape[1] = DIM_MAP["out_img_shape"][1]
        params_dict["conv_shape"] = conv_shape
        params_dict["invalid_data_rm_flag"] = int(invalid_data_rm_flag)
        with tvm.tag_scope('conv_vector_remove_pad'):
            res_tensor = tvm.compute(conv_shape,
                                     lambda batch, cout1, howo, cout0:
                                     res(0 if group == 1 else cout1 // cout1_opt,
                                         batch,
                                         cout1 if group == 1 else cout1 % cout1_opt,
                                         howo, cout0),
                                     name='remove_pad' + "_cc_" + str(NAME_INDEX[0]), attrs=params_dict)
        return res_tensor

    def remove_pad_fp16_dsl(res, conv_shape, invalid_data_rm_flag):
        """
        res_c
        Parameters
        ----------
        res: input tensor.

        conv_shape: res_c true shape

        invalid_data_rm_flag: True for conv2d without removing pad.

        Returns
        -------
        res_c tensor
        """
        if invalid_data_rm_flag:
            res_c = tvm.compute(res.shape,
                                lambda batch, cout1, howo, cout0:
                                res(batch, cout1, howo, cout0),
                                name='invalid_conv2d_rmpad')
        else:
            res_c = tvm.compute(conv_shape,
                                lambda batch, cout1, howo, cout0:
                                res(batch, cout1, howo, cout0),
                                name='C',
                                tag=OP_TAG + "C",
                                attrs={"width_out": ConvParam.w_out, "conv_shape": conv_shape})
            ConvParam.tensor_map["C"] = res_c
        return res_c

    def remove_padded_column(padded_tensor, res_shape):
        """
        remove padded column when v200_width_out_1_flag is True
        Parameters
        ----------
        padded_tensor: input tensor with wout=2

        res_shape: res_shape with wout=1

        Returns
        -------
        res_tensor with wout=1
        """
        res_tensor = tvm.compute(res_shape,
                                 lambda batch, cout1, howo, cout0:
                                 padded_tensor(batch, cout1, howo*2, cout0),
                                 name='remove_padded_column',
                                 tag=OP_TAG + 'remove_padded_column',
                                 attrs={"width_out": ConvParam.w_out})
        ConvParam.tensor_map["remove_padded_column"] = res_tensor
        return res_tensor

    def check_optim_dict(optim_dict, para_dict, data, weight):
        """
        check optim dict
        Parameters
        ----------
        optim_dict: optim dict

        para_dict: para dict

        data: fmap

        weight: filter

        Returns
        -------
        check function
        """
        if isinstance(optim_dict, dict) and "c0_optim_flg" in optim_dict.keys() and \
                isinstance(optim_dict["c0_optim_flg"], bool):
            pass
        else:
            err_man.raise_err_specific("conv2d", "Invalid optim_dict check")

        kernel_one_one = (para_dict["filter_h"] == 1) and (para_dict["filter_w"] == 1)
        if optim_dict["c0_optim_flg"]:
            c0_value = _fmap_c0_check_value(weight.dtype, optim_dict)
            if weight.dtype != "int8" and data.shape[1].value == 1 and data.shape[4].value == c0_value \
                    and weight.shape[3].value == 16 and not kernel_one_one:
                pass
            else:
                err_man.raise_err_specific("conv2d", "Invalid config for c0=4 optimize feature.")

    def check_data(data, optim_dict):
        """
        Check conv fmap param.
        """
        if not isinstance(data, tvm.tensor.Tensor):
            err_man.raise_err_specific("conv2d", "the first Input parameter must be a tvm.tensor.Tensor")
        if len(data.shape) != 5:
            err_man.raise_err_specific("conv2d", "the first Input parameter must be a 5 dim tvm.tensor.Tensor")
        check_dtype_list = ('int8', "float16")
        util.check_dtype_rule(data.dtype, check_dtype_list)

        block_size_k = 4 if optim_dict["c0_optim_flg"] and is_support_v200() and optim_dict["use_v200_c04_flg"] \
            else CUBE_MKN[data.dtype]['mac'][1]
        if data.shape[4].value != block_size_k:
            err_man.raise_err_scene_equal_limitation("conv2d",
                                                     "the last dim of first Input parameter", str(block_size_k))

    def check_weight(weight):
        """
        Check conv weight param.
        """
        if not isinstance(weight, tvm.tensor.Tensor):
            err_man.raise_err_specific("conv2d", "the first Input parameter must be a tvm.tensor.Tensor")
        if len(weight.shape) != 4:
            err_man.raise_err_specific("conv2d", "the first Input parameter must be a 4 dim tvm.tensor.Tensor")
        check_dtype_list = ('int8', "float16")

        util.check_dtype_rule(weight.dtype, check_dtype_list)
        block_size_k = CUBE_MKN[weight.dtype]['mac'][1]

        if weight.shape[3].value != block_size_k:
            err_man.raise_err_scene_equal_limitation("conv2d",
                                                     "the last dim of first Input parameter", str(block_size_k))

    def _save_conv_param():
        """
        save conv params in ConvParam.
        """
        def _save_tensor_shape():
            """
            calculate common used shape info and store in ConvParam.dim_map

            fmap shape info:
            fmap_ori_nchw_shape: [batch, cin, hin, win] (cin isn't 16/32 aligned)
            fmap_align_nchw_shape: [batch, cin1*cin0, hin, win]
            fmap_5hd_shape: [batch, cin1, hin, win, cin0]
            fmap_tiling_a_shape: [batch, c1_opt, hin, win, cin0]

            weight shape info:
            weight_ori_nchw_shape: [cout, cin, hk, wk] (cin and cout aren't 16/32 aligned)
            weight_align_nchw_shape: [cout1*cout0, cin1*cin0, hk, wk]
            weight_fracz_shape: [G*c1_opt*hk*wk, cout1_opt, cout0, cin0]
            weight_tiling_b_shape: [cout1_opt*cout0, c1_opt, hk, wk, cin0]

            output shape info:
            output_5hd_shape: [batch, cout1, hout, wout, cout0]
            output_conv_res_shape: [batch, cout1, hout*wout, cout0]
            output_mad_res_shape: [G, batch, cout1_opt, howo_16align, cout0]
            output_tiling_c_shape: [batch, cout1_opt, hout, wout, cout0]

            """
            def _get_dsl_fmap_shape_nc1hwc0():
                """
                Get fmap_shape_nc1hwc0 for dsl interface.

                cin0 is set to 4 when c0_optim enabled

                Returns
                -------
                fmap_shape_nc1hwc0: a list of [batch, cin1, hin, win, cin0]
                """
                if not ConvParam.dynamic_flag:
                    fmap_shape_nc1hwc0 = list(shape_to_list(data.shape))
                else:
                    fmap_shape_nc1hwc0 = list(data.shape)

                if optim_dict["c0_optim_flg"]:
                    fmap_shape_nc1hwc0[4] = 4
                return fmap_shape_nc1hwc0

            # calculate weight shape
            cout_ori, cin_ori, h_k, w_k = para_dict["weight_ori_shape_nchw"]
            gopt_c1opt_hk_wk, cout1_opt, cout0, cin0 = shape_to_list(weight.shape)
            cin1_opt = para_dict["c1_opt"]
            cin_align = ceil(cin_ori, cin0)
            cout_align = ceil(cout_ori, cout0)
            group_opt = para_dict["group_opt"]
            ConvParam.dim_map["weight_ori_nchw_shape"] = [cout_ori, cin_ori, h_k, w_k]
            ConvParam.dim_map["weight_align_nchw_shape"] = [cout_align, cin_align, h_k, w_k]
            ConvParam.dim_map["weight_fracz_shape"] = [gopt_c1opt_hk_wk, cout1_opt, cout0, cin0]
            ConvParam.dim_map["weight_tiling_b_shape"] = [cout1_opt*cout0, cin1_opt, h_k, w_k, cin0]
            # calculate fmap shape
            batch, cin1, hin, win, _ = para_dict["a_shape"]

            ConvParam.dim_map["fmap_ori_nchw_shape"] = [batch, cin_ori, hin, win]
            ConvParam.dim_map["fmap_align_nchw_shape"] = [batch, cin1*cin0, hin, win]
            ConvParam.dim_map["fmap_5hd_shape"] = para_dict["a_shape"]

            dsl_fmap_shape = _get_dsl_fmap_shape_nc1hwc0()
            ConvParam.dim_map["fmap_tiling_a_shape"] = [dsl_fmap_shape[0], cin1_opt,
                                                        dsl_fmap_shape[2], dsl_fmap_shape[3], cin0]

            if optim_dict["c0_optim_flg"]:
                ConvParam.dim_map["fmap_tiling_a_shape"] = [dsl_fmap_shape[0], cin1_opt,
                                                            dsl_fmap_shape[2], dsl_fmap_shape[3], 4]
                ConvParam.dim_map["weight_tiling_b_shape"] = [cout1_opt*cout0, 1, h_k, w_k, 4]
            # calculate mad/output shape
            hout = ConvParam.h_out
            wout = ConvParam.w_out

            ConvParam.dim_map["output_5hd_shape"] = [batch, int_ceil_div(cout_ori, cout0), hout, wout, cout0]
            ConvParam.dim_map["output_conv_res_shape"] = [batch, int_ceil_div(cout_ori, cout0), hout*wout, cout0]
            ConvParam.dim_map["output_mad_res_shape"] = [group_opt, batch, cout1_opt, ceil(hout*wout, 16), cout0]
            ConvParam.dim_map["output_tiling_c_shape"] = [batch, cout1_opt, hout, wout, cout0]

        def _save_params():
            """
            save params into ConvParam
            """
            ConvParam.pad_h = para_dict.get("pad_h")
            ConvParam.pad_w = para_dict.get("pad_w")
            ConvParam.stride_h = para_dict.get("stride_h")
            ConvParam.stride_w = para_dict.get("stride_w")
            ConvParam.dilate_h = para_dict.get("dilate_h")
            ConvParam.dilate_w = para_dict.get("dilate_w")
            ConvParam.filter_h = para_dict.get("filter_h")
            ConvParam.filter_w = para_dict.get("filter_w")
            ConvParam.mad_dtype = para_dict.get("mad_dtype")
            ConvParam.res_dtype = res_dtype
            ConvParam.kernel_name = para_dict.get("kernel_name")
            if 'value' in dir(data.shape[0]):
                ConvParam.batch = data.shape[0].value
            else:
                ConvParam.batch = data.shape[0]
            ConvParam.para_dict = para_dict
            if optim_dict["c0_optim_flg"]:
                ConvParam.para_dict["c1_opt"] = 1

            if 'value' in dir(data.shape[2]):
                ConvParam.h_in = data.shape[2].value
            else:
                ConvParam.h_in = data.shape[2]
            if 'value' in dir(data.shape[3]):
                ConvParam.w_in = data.shape[3].value
            else:
                ConvParam.w_in = data.shape[3]

            ConvParam.padding = [ConvParam.pad_h[0], ConvParam.pad_h[1],
                                 ConvParam.pad_w[0], ConvParam.pad_w[1]]

            if ConvParam.padding != [0, 0, 0, 0]:
                ConvParam.has_padding = True

            filter_h_dilation = (ConvParam.filter_h - 1)*ConvParam.dilate_h + 1
            filter_w_dilation = (ConvParam.filter_w - 1)*ConvParam.dilate_w + 1
            if 'ho' in ConvParam.dyn_var_map:
                ConvParam.h_out = ConvParam.dyn_var_map.get('ho')
            else:
                ConvParam.h_out = (ConvParam.h_in + (ConvParam.pad_h[0] + ConvParam.pad_h[1]) -
                                   filter_h_dilation) // ConvParam.stride_h + 1

            if 'wo' in ConvParam.dyn_var_map:
                ConvParam.w_out = ConvParam.dyn_var_map.get('wo')
            else:
                ConvParam.w_out = (ConvParam.w_in + (ConvParam.pad_w[0] + ConvParam.pad_w[1]) -
                                   filter_w_dilation) // ConvParam.stride_w + 1

        def _config_tiling_query_param():
            """
            config ConvParam.tiling_query_param, to be used in info_dict
            """

            ConvParam.tiling_query_param.update(
                {"fmap_shape_nc1hwc0": ConvParam.dim_map["fmap_tiling_a_shape"], # left matrix size of one g_opt(5hd)
                 "shape_w_nc1hwc0": ConvParam.dim_map["weight_tiling_b_shape"], # right matrix size of one g_opt(5hd)
                 "in_dtype": in_dtype, # result matrix size of one group_opt in 5hd
                 "w_dtype": w_dtype,
                 "res_dtype": res_dtype,
                 "mad_dtype": mad_dtype,
                 "padw": para_dict["pad_w"],
                 "padh": para_dict["pad_h"],
                 "strideh": stride_h,
                 "stridew": stride_w,
                 "dilateh": dilate_h,
                 "dilatew": dilate_w,
                 "bias_flag": bias_tensor_flag})
        _save_params()
        _save_tensor_shape()
        _config_tiling_query_param()

    def load2d_to_load3d_flag_set():
        """
        Set a flag for load2d to instead the load3d and
        some checks do not apply to conv1D.
        """
        load2d_to_load3d_flag = (para_dict.get("filter_h") == 1) and \
                                (para_dict.get("filter_w") == 1) and \
                                (pad_top == 0) and (pad_bottom == 0) and \
                                (pad_left == 0) and (pad_right == 0) and \
                                (para_dict.get("stride_h") == 1) and \
                                (para_dict.get("stride_w") == 1) and \
                                (data.dtype == "float16") and \
                                (weight.dtype == "float16")

        _, cin_ori, _, _ = para_dict["weight_ori_shape_nchw"]
        _, _, _, cin0 = shape_to_list(weight.shape)
        if para_dict.get("group_opt") > 1 and is_support_v200() and \
            (para_dict.get("group_opt")*para_dict.get("c1_opt")*cin0 != cin_ori):
            load2d_to_load3d_flag = False
        return load2d_to_load3d_flag

    def _conv1d_split_w_flag_set():
        """
        Set this flag to define whether is doing conv1d.
        """
        h_dynamic_flag = ConvParam.dynamic_flag and get_te_var("fmap_h")
        if not h_dynamic_flag and data.shape[2].value == 1 \
            and para_dict.get("filter_h") == 1 \
            and (pad_top + pad_bottom) == 0 and \
            not load2d_to_load3d_flag:
            ConvParam.conv1d_split_w_flag = True

    def _v200_width_out_1_flag_set():
        """
        special supporting of height_out!=1 && width_out=1 case for v200 soc
        """
        hk_dilation = (para_dict["filter_h"] - 1)*para_dict["dilate_h"] + 1
        wk_dilation = (para_dict["filter_w"] - 1)*para_dict["dilate_w"] + 1
        if 'value' in dir(data.shape[2]):
            h_out = (data.shape[2].value + (para_dict["pad_h"][0] + para_dict["pad_h"][1]) - hk_dilation) // \
                para_dict["stride_h"] + 1
        else:
            h_out = (data.shape[2] + (para_dict["pad_h"][0] + para_dict["pad_h"][1]) - hk_dilation) // \
                para_dict["stride_h"] + 1
        if 'value' in dir(data.shape[3]):
            w_out = (data.shape[3].value + (para_dict["pad_w"][0] + para_dict["pad_w"][1]) - wk_dilation) // \
                para_dict["stride_w"] + 1
        else:
            w_out = (data.shape[3] + (para_dict["pad_w"][0] + para_dict["pad_w"][1]) - wk_dilation) // \
                para_dict["stride_w"] + 1
        l0a_load2d_flag = False
        if list(para_dict["pad_h"]) == [0, 0] \
            and list(para_dict["pad_w"]) == [0, 0] \
            and para_dict["stride_h"] == 1 and para_dict["stride_w"] == 1 \
            and weight.dtype == "float16" \
            and (para_dict["filter_h"]*para_dict["filter_w"] == 1):
            l0a_load2d_flag = True

        if ConvParam.fusion_para["l1_fusion_type"] in (0, 1) or ConvParam.fusion_para["input_memory_type"][0] == 1:
            l0a_load2d_flag = False

        if not check_load3d_w_out_1_support() and h_out != 1 and w_out == 1 and not l0a_load2d_flag:
            para_dict["pad_w"][1] += para_dict["stride_w"] # increasing pad right, N*1 -> N*2
            ConvParam.v200_width_out_1_flag = True
        else:
            ConvParam.v200_width_out_1_flag = False

    def _save_tiling_info_dict(shape_fmap_nc1hwc0, shape_w_nc1hwc0, c_ub_shape, in_dtype, w_dtype, res_dtype,
                               bias_flag, kernel_name):
        """
        Save tiling_info_dict for dynamic.
        """
        if ConvParam.dynamic_flag:
            in_size_h = shape_fmap_nc1hwc0[2]
            in_size_w = shape_fmap_nc1hwc0[3]
            kernel_h = shape_w_nc1hwc0[2]
            kernel_w = shape_w_nc1hwc0[3]
            pad_h = para_dict.get("pad_h")
            pad_w = para_dict.get("pad_w")
            stride_h = para_dict.get("stride_h")
            stride_w = para_dict.get("stride_w")
            dilate_h = para_dict.get("dilate_h")
            dilate_w = para_dict.get("dilate_w")
            hk_dilation = (kernel_h - 1)*dilate_h + 1
            wk_dilation = (kernel_w - 1)*dilate_w + 1
            if isinstance(pad_h, int):
                pad_h = (pad_h, pad_h)
            if isinstance(pad_w, int):
                pad_w = (pad_w, pad_w)
            w_out = (in_size_w + (pad_w[0] + pad_w[1] - wk_dilation)) // stride_w + 1
            h_out = (in_size_h + (pad_h[0] + pad_h[1] - hk_dilation)) // stride_h + 1
            c_shape = [c_ub_shape[0], c_ub_shape[1], h_out, w_out, c_ub_shape[3]]
            ConvParam.tiling_info_dict = {
                "op_type": 'conv2d',
                "a_shape": list(shape_fmap_nc1hwc0),
                "placeholder_fmap_5hd_shape": list(ConvParam.dim_map["fmap_5hd_shape"]),
                "b_shape": list(shape_w_nc1hwc0),
                "c_shape": c_shape,
                "a_dtype": in_dtype,
                "b_dtype": w_dtype,
                "c_dtype": res_dtype,
                "mad_dtype": mad_dtype,
                "pad": [pad_w[0], pad_w[1], pad_h[0], pad_h[1]],
                "stride": [stride_h, stride_w],
                "dilation": [dilate_h, dilate_w],
                "group": 1,
                "bias_flag": bias_flag,
                "fused_coefficient": [0, 0, 0],
                "fused_channel_wise": [0, 0, 0],
                "in_fm_memory_type": [],
                "out_fm_memory_type": [],
                "l1_fusion_type": -1,
                "fusion_type": 0,
                "reserved_ub": 0,
                "kernel_name": kernel_name,
                "dynamic_shape_flag": True}

    def _get_dynamic_para():
        """
        Get dynamic para.
        """

        fmap_range = []
        dyn_var_map = {}
        is_dynamic = False
        if isinstance(data.shape[0], tvm.expr.Var):
            fmap_range.append(get_te_var("batch_n").get_bound())
            dyn_var_map["batch_n"] = get_te_var("batch_n").get_tvm_var()
        else:
            fmap_range.append((data.shape[0], data.shape[0]))

        if isinstance(data.shape[2], tvm.expr.Var):
            fmap_range.append(get_te_var("fmap_h").get_bound())
            dyn_var_map["fmap_h"] = get_te_var("fmap_h").get_tvm_var()
            dyn_var_map["ho"] = get_te_var("ho").get_tvm_var()
        else:
            fmap_range.append((data.shape[2], data.shape[2]))

        if isinstance(data.shape[3], tvm.expr.Var):
            fmap_range.append(get_te_var("fmap_w").get_bound())
            dyn_var_map["fmap_w"] = get_te_var("fmap_w").get_tvm_var()
            dyn_var_map["wo"] = get_te_var("wo").get_tvm_var()
        else:
            fmap_range.append((data.shape[3], data.shape[3]))
        if dyn_var_map:
            is_dynamic = True

        dynamic_para = {
            "fmap_range": fmap_range,
            "var_map": dyn_var_map,
            "correct_range_flag": para_dict.get("correct_range_flag", False),
            "new_in_range": para_dict.get("new_in_range")
        }
        return dynamic_para, is_dynamic

    def calculate_remove_pad_params(padded_column_shape, v200_width_out_1_flag):
        """
        for v200 quant fusion, when hout!=1 && wout=1, info is passed to next op
        remove padded column is done by next op
        """
        true_conv_shape = []
        for i in padded_column_shape:
            true_conv_shape.append(i)
        if v200_width_out_1_flag:
            true_conv_shape[-2] = true_conv_shape[-2] // 2
        true_conv_shape = tuple(true_conv_shape)
        params_dict = {"remove_padded_column_in_next_op": ConvParam.v200_width_out_1_flag,
                       "true_conv_shape": true_conv_shape}
        return params_dict

    def _prefusion_identify():
        """
        set corresponding flags for prefusion like relu/aipp/strided_read
        """
        if "relu" in data.op.name:
            ConvParam.pre_relu_flag = True

        if data.op.tag == "strided_read":
            ConvParam.strided_read_flag = True

        if data.op.tag == "aipp_res_convolution":
            ConvParam.aipp_fuse_flag = True

    def _save_input_tensor():
        """
        save conv input tensors into TENSOR_MAP
        """

        # save weight
        TENSOR_MAP["filter"] = weight
        # save fmap
        if ConvParam.strided_read_flag or ConvParam.aipp_fuse_flag:
            TENSOR_MAP["fmap"] = data.op.input_tensors[0]  # fmap stands for the data in out memory
        else:
            TENSOR_MAP["fmap"] = data

        # save bias
        if bias_tensor_flag:
            if data.dtype != "int8":
                TENSOR_MAP["fp16_bias"] = bias_tensor   # float bias
            else:
                TENSOR_MAP["int32_bias"] = bias_tensor # quant bias

    def _input_parameters_completion_and_modification(dsl_flag, optim_dict):
        """
        complete required parameters and modify parameters for special scenes

        """

        if optim_dict is None:
            optim_dict = {"c0_optim_flg": False, "use_v200_c04_flg": False}

        if not isinstance(para_dict, dict):
            err_man.raise_err_check_type("conv2d", "the third Input", "dict", "not dict")

        if "mad_dtype" not in para_dict:
            if weight.dtype == "int8":
                mad_dtype = "int32"
            elif get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
                mad_dtype = "float16"
            else:
                mad_dtype = "float32"
            para_dict["mad_dtype"] = mad_dtype

        if "offset_x" not in para_dict:
            para_dict["offset_x"] = 0

        if "kernel_name" not in para_dict:
            para_dict["kernel_name"] = "conv2d"

        if "pad_h" not in para_dict:
            if "padh" in para_dict:
                para_dict["pad_h"] = para_dict["padh"]
            else:
                err_man.raise_err_specific_input_shape("conv2d", "para_dict must contain pad_h")
        if "pad_w" not in para_dict:
            if "padw" in para_dict:
                para_dict["pad_w"] = para_dict["padw"]
            else:
                err_man.raise_err_specific_input_shape("conv2d", "para_dict must contain pad_w")
        if "stride_h" not in para_dict:
            if "strideh" in para_dict:
                para_dict["stride_h"] = para_dict["strideh"]
            else:
                err_man.raise_err_specific_input_shape("conv2d", "para_dict must contain stride_h")
        if "stride_w" not in para_dict:
            if "stridew" in para_dict:
                para_dict["stride_w"] = para_dict["stridew"]
            else:
                err_man.raise_err_specific_input_shape("conv2d", "para_dict must contain stride_w")
        if "dilate_h" not in para_dict:
            if "dilateh" in para_dict:
                para_dict["dilate_h"] = para_dict["dilateh"]
            else:
                para_dict["dilate_h"] = 1
        if "dilate_w" not in para_dict:
            if "dilatew" in para_dict:
                para_dict["dilate_w"] = para_dict["dilatew"]
            else:
                para_dict["dilate_w"] = 1
        if "filter_h" not in para_dict:
            if "filterh" in para_dict:
                para_dict["filter_h"] = para_dict["filterh"]
            else:
                err_man.raise_err_specific_input_shape("conv2d", "para_dict must contain filter_h")
        if "filter_w" not in para_dict:
            if "filterw" in para_dict:
                para_dict["filter_w"] = para_dict["filterw"]
            else:
                err_man.raise_err_specific_input_shape("conv2d", "para_dict must contain filter_w")

        pad_h = para_dict["pad_h"]
        pad_w = para_dict["pad_w"]
        if isinstance(pad_h, int):
            para_dict["pad_h"] = [pad_h, pad_h]
        if isinstance(pad_w, int):
            para_dict["pad_w"] = [pad_w, pad_w]


        # modification of input parameters only happens here.
        if ConvParam.pre_relu_flag:
            dsl_flag = False

        return dsl_flag, optim_dict

    def _handle_fp16_bias_add(conv_res, bias_tensor, bias_tensor_flag, dsl_flag):
        """
        calculate res of bias_add in fp16
        """
        if bias_tensor_flag and ((not ConvParam.dynamic_flag) or (ConvParam.dynamic_flag and not dsl_flag)):
            res = bias_add(conv_res, bias_tensor)
            return res
        return conv_res

    ConvParam.set_default()
    ConvParam.dynamic_para, ConvParam.dynamic_flag = _get_dynamic_para()
    ConvParam.dyn_var_map = ConvParam.dynamic_para.get("var_map")

    bias_tensor = para_dict.get("bias_tensor")
    bias_tensor_flag = isinstance(bias_tensor, tvm.tensor.Tensor)

    # identify prefusion (such as relu/aipp/strided_read) and save into ConvParam
    _prefusion_identify()

    # save input tensor into ConvParam
    _save_input_tensor()

    # complete required keys of para_dict and necessary check
    # input parameters can only be modified here
    dsl_flag, optim_dict = _input_parameters_completion_and_modification(dsl_flag, optim_dict)

    in_dtype = data.dtype
    w_dtype = weight.dtype

    stride_h = para_dict["stride_h"]
    stride_w = para_dict["stride_w"]
    dilate_h = para_dict["dilate_h"]
    dilate_w = para_dict["dilate_w"]
    filter_h = para_dict["filter_h"]
    filter_w = para_dict["filter_w"]
    pad_top = para_dict["pad_h"][0]
    pad_bottom = para_dict["pad_h"][1]
    pad_left = para_dict["pad_w"][0]
    pad_right = para_dict["pad_w"][1]
    offset_x = para_dict["offset_x"]
    kernel_name = para_dict["kernel_name"]
    mad_dtype = para_dict["mad_dtype"]
    res_dtype = "float16"
    if (in_dtype, w_dtype) == ("int8", "int8"):
        res_dtype = "int32"
    #====================fetch L1fusion information from pass interface=============
    l1_fusion_enable_flag = get_current_build_config("enable_L1_fusion")
    l2_fusion_enable_flag = get_current_build_config("enable_L2_fusion") and get_current_build_config("l2_mode") == 1
    lxfusion_enable_flag = l1_fusion_enable_flag or l2_fusion_enable_flag

    if lxfusion_enable_flag: # lxfusion enabled
        buffer_manager = get_buffer_manager()
        # set info to buffer_manager
        if not dsl_flag:
            if bias_tensor_flag:
                tensor_list = [data, weight, bias_tensor, None]
            else:
                tensor_list = [data, weight, None]

            buffer_manager.set_tensor_list(tensor_list)

        l1_fusion_type = buffer_manager.get_l1_fusion_type()
        input_info = buffer_manager.get_tensor_info(data)
        if input_info:
            input_scope = input_info.get_buffer_scope()
        else:
            input_scope = "global"

        if input_scope == "global":
            ConvParam.fusion_para["input_memory_type"].append(DDR_SCOPE) # 0 from DDR 1 from L1
        elif input_scope == "local.L1_Fusion":
            if l1_fusion_type == -1:
                err_man.raise_err_specific(
                    "conv2d", "input buffer scope must be global when l1_fusion_type is -1.")
            if optim_dict["c0_optim_flg"]:
                err_man.raise_err_specific(
                    "conv2d", "fmap from L1 is not supported in c04 optimization.")
            ConvParam.fusion_para["input_memory_type"].append(L1_FUSION_SCOPE)
        else:
            err_man.raise_err_specific("conv2d", "input buffer scope must be global or local.L1_Fusion.")

        if (l2_fusion_enable_flag or (not l1_fusion_enable_flag)) and \
                (input_scope == "local.L1_Fusion" or l1_fusion_type != -1):
            err_man.raise_err_specific_user(
                "conv2d", "if enable L2 fusion or not enable L1 fusion, " +
                "l1_fusion_type must be -1 and input buffer scope cannot be local.L1_Fusion.")

        ConvParam.fusion_para["l1_fusion_type"] = l1_fusion_type
    else: # lxfusion disabled
        ConvParam.fusion_para["input_memory_type"].append(DDR_SCOPE)

    ConvParam.fusion_para["fmap_l1_addr_flag"] = para_dict["fusion_para"].get("fmap_l1_addr_flag", "nothing")
    ConvParam.fusion_para["fmap_l1_valid_size"] = para_dict["fusion_para"].get("fmap_l1_valid_size", -1)
    ConvParam.fusion_para["lxfusion_enable_flag"] = lxfusion_enable_flag
    #===========================================================================================
    _v200_width_out_1_flag_set()
    load2d_to_load3d_flag = load2d_to_load3d_flag_set()
    _conv1d_split_w_flag_set()

    check_optim_dict(optim_dict, para_dict, data, weight)
    check_data(data, optim_dict)
    check_weight(weight)

    _save_conv_param()

    shape_in = ConvParam.dim_map["fmap_align_nchw_shape"]
    shape_w = ConvParam.dim_map["weight_align_nchw_shape"]
    shape_fmap_nc1hwc0 = ConvParam.dim_map["fmap_tiling_a_shape"]
    shape_w_nc1hwc0 = ConvParam.dim_map["weight_tiling_b_shape"]
    check_conv_shape(shape_in, shape_w, pad_top, pad_bottom, pad_left, pad_right, stride_h, stride_w, in_dtype, w_dtype,
                     optim_dict, dilateh=dilate_h, dilatew=dilate_w,
                     dynamic_para=ConvParam.dynamic_para, groups=para_dict['group'])

    # conv1d and dma im2col case, conv1d use dma
    if ConvParam.l0a_dma_flag:
        ConvParam.conv1d_split_w_flag = False

    conv_shape = ConvParam.dim_map["output_conv_res_shape"]
    if "invalid_data_rm" not in optim_dict:
        optim_dict["invalid_data_rm"] = False
    invalid_data_rm_flag = optim_dict["invalid_data_rm"]
    ConvParam.invalid_data_rm_flag = invalid_data_rm_flag

    if in_dtype == "int8":  # quant
        if dsl_flag:  # quant fusion
            conv_res = _cube_compute(data, weight, mad_dtype,
                                     tiling=ConvParam.tiling, optim_dict=optim_dict, bias=bias_tensor)
            remove_pad_params = calculate_remove_pad_params(conv_shape, ConvParam.v200_width_out_1_flag)
            res_remove_pad = remove_pad_quant_dsl(conv_res, conv_shape, invalid_data_rm_flag,
                                                  params_dict=remove_pad_params)
            _save_tiling_info_dict(shape_fmap_nc1hwc0, shape_w_nc1hwc0, list(res_remove_pad.shape),
                                   in_dtype, w_dtype, res_dtype, bias_tensor_flag, kernel_name)
            return res_remove_pad
        # quant single op
        conv_res = _cube_compute(data, weight, mad_dtype,
                                 tiling=ConvParam.tiling, optim_dict=optim_dict, bias=bias_tensor)
        res = _quant_l0c2ub_compute(conv_res, res_dtype)
        if ConvParam.v200_width_out_1_flag:
            remove_padded_column_shape = conv_shape.copy()
            remove_padded_column_shape[-2] = remove_padded_column_shape[-2] // 2
            res = remove_padded_column(res, remove_padded_column_shape)
    else:  # float
        no_vector_flag = (not dsl_flag) and (not bias_tensor_flag) # no vec calculation in UB
        conv_res = cub_fp16_compute(data, weight, mad_dtype, res_dtype, stride_h, stride_w,
                                    dilate_h, dilate_w, filter_h, filter_w,
                                    bias=False, no_vector=no_vector_flag, tiling=ConvParam.tiling,
                                    conv_fused_flag=dsl_flag, optim_dict=optim_dict, kernel_name=kernel_name,
                                    padding_mode=para_dict.get("padding_mode"),
                                    pooling_mode=para_dict.get("pooling_mode"))
        res = conv_res

        if ConvParam.v200_width_out_1_flag:
            remove_padded_column_shape = conv_shape.copy()
            remove_padded_column_shape[-2] = remove_padded_column_shape[-2] // 2
            conv_res = remove_padded_column(conv_res, remove_padded_column_shape)
            res = conv_res

        res = _handle_fp16_bias_add(conv_res, bias_tensor, bias_tensor_flag, dsl_flag)

    _save_tiling_info_dict(shape_fmap_nc1hwc0, shape_w_nc1hwc0, list(res.shape),
                           in_dtype, w_dtype, res_dtype, bias_tensor_flag, kernel_name)

    if ConvParam.v200_width_out_1_flag:
        conv_shape[-2] = conv_shape[-2] // 2

    if dsl_flag:
        res_c = remove_pad_fp16_dsl(res, conv_shape, invalid_data_rm_flag)
        if ConvParam.dynamic_flag and bias_tensor_flag:
            res_c = bias_add(res_c, bias_tensor)
        return res_c

    res_remove_pad = remove_pad(res, conv_shape)

    if lxfusion_enable_flag:
        tensor_list[-1] = res_remove_pad
        buffer_manager.set_tensor_list(tensor_list)

    return res_remove_pad
