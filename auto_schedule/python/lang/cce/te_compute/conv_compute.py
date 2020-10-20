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
from te import tvm
from te.platform import cce_conf
from te.platform import CUBE_MKN
from topi.cce import util
from topi.cce.util import check_load3d_w_out_1_support
from te.utils.error_manager import error_manager_conv2d as err_man
from te.platform.operation import get_te_var

# fmapH, fmapW must be in [1,4096]
FMAP_HW_MIN = 1
FMAP_W_MAX = 4096
FMAP_H_MAX = 100000

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


def is_support_v200():
    """
    Check if Ascend610/Ascend615/Ascend710/Hi3796CV300CS version.
    ----------

    Returns
    -------
    True:  Ascend610/Ascend615/Ascend710/Hi3796CV300CS version
    False: Other version
    """
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    if soc_version in ("Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS"):
        return True
    return False


def check_conv_shape(shape_in, shape_w, pad_top, pad_bottom,
                     pad_left, pad_right, strideh, stridew, in_dtype, w_dtype, fusion_para,
                     optim_dict=None, dilateh=1, dilatew=1, dynamic_para=None):
    """

    Parameters
    ----------
    shape_in : shape of data_in

    shape_w : shape of filter

    padh: the padding shape in H

    padw: the padding shape in weight

    strideh: the stride value in H

    stridew: the stride value in weight

    dilateh: the dilate value in H

    dilatew: the dilate value in weight

    optim_dict: optimize feature dict

    in_dtype : the feature map data type

    w_dtype : the weight data type

    fusion_para: the config for Lx Fusion

    Returns
    -------
    None

    """

    def fusion_para_check(fusion_para, pad_top, pad_bottom, shape_in):
        """
        Check Lx fusion para.
        """
        def handle_valid_shape():
            """
            Check valid shape in lx fusion para.
            """
            if not slice_offset:
                err_man.raise_err_specific("conv2d",
                                           "if valid_shape exists, "
                                           + "offset can not be []")
            slice_offset_check_flg = (slice_offset[2] < shape_in[2]) and \
                (slice_offset[2] >= 0) \
                and slice_offset[0] == 0 and slice_offset[1] == 0 and \
                slice_offset[3] == 0 and slice_offset[4] == 0

            valid_shape_check_flg = (valid_shape[2] + slice_offset[2] <=
                                     shape_in[2]) and (valid_shape[2] >= 0) \
                and valid_shape[0] == shape_in[0] and valid_shape[3] == \
                shape_in[3] and valid_shape[1]*valid_shape[4] == shape_in[1]

            if not slice_offset_check_flg:
                err_man.raise_err_check_the_validity_of_one_variable("conv2d",
                                                                     "Invalid valid_shape",
                                                                     str(slice_offset))
            if not valid_shape_check_flg:
                err_man.raise_err_check_the_validity_of_variable("conv2d",
                                                                 "Invalid valid_shape",
                                                                 str(valid_shape), str(shape_in))

            if fusion_para.get("input_memory_type") == 1:
                if slice_offset[2] == 0 and pad_bottom != 0:
                    err_man.raise_err_scene_limitation("conv2d", "first",
                                                       "pad_bottom", "0")
                if slice_offset[2] == (shape_in[2] - valid_shape[2]) and \
                        pad_top != 0:
                    err_man.raise_err_scene_limitation("conv2d", "last",
                                                       "pad_top", "0")
                if (slice_offset[2] > 0 and slice_offset[2] <
                        (shape_in[2] - valid_shape[2])) and \
                        (pad_top != 0 or pad_bottom != 0):
                    err_man.raise_err_scene_limitation("conv2d", "middle",
                                                       "pad_top and pad_bottom", "0")

        if fusion_para is None:
            fusion_para = {"input_memory_type": 0,
                           "output_memory_type": 0,
                           "valid_shape": (),
                           "slice_offset": (),
                           "l1_fusion_type": -1,
                           "fmap_l1_addr_flag": False,
                           "fmap_l1_valid_size": -1}

        valid_shape = fusion_para.get("valid_shape")
        slice_offset = fusion_para.get("slice_offset")
        l1_fusion_type = fusion_para.get("l1_fusion_type")
        input_memory_type = fusion_para.get("input_memory_type")
        output_memory_type = fusion_para.get("output_memory_type")

        if l1_fusion_type == -1:
            if input_memory_type == 1 or output_memory_type == 1:
                err_man.raise_err_check_the_validity_of_variable("conv2d",
                                                                 "input_memory_type/output_memory_type"
                                                                 + " must be 0 when l1_fusion_type is -1",
                                                                 str(input_memory_type), str(output_memory_type))

        if valid_shape:
            handle_valid_shape()

    def _l1_buffer_size_check(max_feature_map_l1, fusion_para, dynamic_mode=None):
        """
        Check for not bigger than L1 size.
        """
        l1_buffer_size = cce_conf.get_soc_spec("L1_SIZE")
        l1_fusion_type = fusion_para.get("l1_fusion_type")
        if l1_fusion_type in (0, 1):
            pass
        elif int(max_feature_map_l1) > l1_buffer_size:
            if dynamic_mode is None:
                err_man.raise_err_specific("conv2d",
                                           "Input is too large, "
                                           + "the minimum tiling may exceed L1_Buffer")
            else:
                err_man.raise_err_specific("conv2d",
                                           "Input range is too large, "
                                           + "the minimum tiling may exceed L1_Buffer")

    def conv1d_split_w_flag_set():
        """
        For load2d case and load3d cases, set a conv1d_split_w_flag and
        some checks do not apply to conv1D
        """
        conv1d_split_w_flag = shape_in[2] == 1 and shape_w[2] == 1 \
        and pad_top == 0 and pad_bottom == 0
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
        if dynamic_mode and (dilateh != 1 or dilatew != 1):
            err_man.raise_err_specific_input_shape("conv2d",
                                                   "Invalid dilation check")
        if dilateh < DILATE_MIN or dilateh > DILATE_MAX:
            range_value = "".join([str(DILATE_MIN), ", ", str(DILATE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                 "dilateh", str(dilateh))
        if dilatew < DILATE_MIN or dilatew > DILATE_MAX:
            range_value = "".join([str(DILATE_MIN), ", ", str(DILATE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                 "dilatew", str(dilatew))

    conv1d_split_w_flag = conv1d_split_w_flag_set()
    dynamic_mode = None if dynamic_para is None \
        else dynamic_para.get("dynamic_mode")
    fmap_range = None if dynamic_para is None \
        else dynamic_para.get("fmap_range")

    def check_fm_w_flag_set():
        """
        Check fmap split width flag.
        """
        check_fm_w_flag = False
        check_fm_w_flag = (int(shape_in[3]) < FMAP_HW_MIN or int(shape_in[3]) > FMAP_W_MAX) \
            and not conv1d_split_w_flag
        return check_fm_w_flag

    def _check_fmap_range():
        """
        Check fmap range.
        """
        if dynamic_mode not in ("dynamic_hw", "dynamic_all"):
            if int(shape_in[2]) < FMAP_HW_MIN or int(shape_in[2]) > FMAP_H_MAX:
                range_value = "".join([str(FMAP_HW_MIN), ", ", str(FMAP_H_MAX)])
                err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                     "feature map H", shape_in[2])
            if check_fm_w_flag_set():
                range_value = "".join([str(FMAP_HW_MIN), ", ", str(FMAP_W_MAX)])
                err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                     "feature map W", shape_in[3])
            if conv1d_split_w_flag and \
            (shape_in[3] < FMAP_W_MIN_SPLIT_W or shape_in[3] > \
                FMAP_W_MAX_SPLIT_W):
                range_value = "".join([str(FMAP_W_MIN_SPLIT_W), ", ", str(FMAP_W_MAX_SPLIT_W)])
                err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                     "feature map W when split w", shape_in[3])
        elif dynamic_mode == "dynamic_hw":
            range_value = "".join([str(FMAP_HW_MIN), ", ", str(FMAP_H_MAX)])
            if int(fmap_range[0][0]) < FMAP_HW_MIN or int(fmap_range[0][1]) > FMAP_H_MAX:
                err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                     "feature map H's range", fmap_range[0][1])
            range_value = "".join([str(FMAP_HW_MIN), ", ", str(FMAP_W_MAX)])
            if int(fmap_range[1][0]) < FMAP_HW_MIN or int(fmap_range[1][1]) > FMAP_W_MAX:
                err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                     "feature map W's range", fmap_range[1][1])

    _check_fmap_range()
    if dynamic_mode is None:
        util.check_shape_rule(shape_in, CONV_SHAPE_DIM, CONV_SHAPE_DIM)
    util.check_shape_rule(shape_w, CONV_SHAPE_DIM, CONV_SHAPE_DIM)

    if shape_in[1] != shape_w[1]:
        err_man.raise_err_scene_equal_limitation("conv2d",
                                                 "input feature map channel", "filter channel")

    if optim_dict is None:
        optim_dict = {"c0_optim_flg": False, "use_v200_c04_flg": False}
    block_size_k = CUBE_MKN[in_dtype]['mac'][1]
    shape_in[1] = ((shape_in[1] + block_size_k - 1) //
                   block_size_k)*block_size_k
    # int8 feature_map_channel_in is aligned by 16, but weight_channel_in is aligned by 32.
    shape_w[1] = ((shape_in[1] + block_size_k - 1) //
                  block_size_k)*block_size_k
    if optim_dict["c0_optim_flg"]:
        shape_in[1] = 4
        shape_w[1] = 4
    h_i = shape_in[2]
    w_i = shape_in[3]
    h_k = shape_w[2]
    w_k = shape_w[3]
    if fusion_para and fusion_para.get("valid_shape"):
        h_i = fusion_para.get("valid_shape")[2]

    # dilateh, dilatew check
    dilate_check()

    hk_dilation = (h_k - 1)*dilateh + 1
    wk_dilation = (w_k - 1)*dilatew + 1

    # calculated by h_i and w_i
    h_out = (h_i + (pad_top + pad_bottom) - hk_dilation) // strideh + 1
    # calculated by h_i and w_i
    w_out = (w_i + (pad_left + pad_right) - wk_dilation) // stridew + 1

    def _check_load3d_constraint(h_i, w_i, h_out, w_out):
        """
        Check load3d constraint.
        """
        load2d_pass_flag = (h_k == 1) and (w_k == 1) and \
                           (pad_top == 0) and (pad_bottom == 0) and \
                           (pad_left == 0) and (pad_right == 0) and \
                           (strideh == 1) and (stridew == 1)
        # w_out = 1 case only support load2d or (chips in [Ascend310,
        # Hi3796CV300CS] and fmap_w with padding equals
        # filters_w after dilation
        hout_equal_1 = h_i + pad_top + pad_bottom - hk_dilation == 0
        wout_equal_1 = w_i + pad_left + pad_right - wk_dilation == 0
        wout_equal_1_pass_flag = wout_equal_1 \
            if check_load3d_w_out_1_support() else load2d_pass_flag
        # Ascend910 supports w_out equals 1 and h_out equals 1
        out_both_equal_1_pass_flag = hout_equal_1 and wout_equal_1

        if int(h_out) < 1 or int(w_out) < 1:
            err_man.raise_err_specific("conv2d",
                                       "output shape should greater than 0, " +
                                       "please check input shape\n")
        elif int(w_out) == 1:
            if not (wout_equal_1_pass_flag or out_both_equal_1_pass_flag):
                err_man.raise_err_specific_input_shape("conv2d",
                                                       "op [Conv2D] output featuremap w == 1, " +
                                                       "the input parameter must follow rule: " +
                                                       "chips_version in [Ascend310, Hi3796CV300CS] and " +
                                                       "fmap_h(with padding) == filters_h(after dilation)")
        else:
            pass

    def _check_pad():
        """
        Check pad.
        """
        # padh, padw check
        if pad_top < PAD_MIN or pad_bottom < PAD_MIN or \
                pad_top > PAD_MAX or pad_bottom > PAD_MAX:
            range_value = "".join([str(PAD_MIN), ", ", str(PAD_MAX)])
            actual_value = "".join([str(pad_top), ", ", str(pad_bottom)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                 "pad_top or pad_bottom", actual_value)
        if pad_left < PAD_MIN or pad_right < PAD_MIN or \
                pad_left > PAD_MAX or pad_right > PAD_MAX:
            range_value = "".join([str(PAD_MIN), ", ", str(PAD_MAX)])
            actual_value = "".join([str(pad_left), ", ", str(pad_right)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                 "pad_left or pad_right", actual_value)

    def _check_dynamic_range():
        """
        Check dynamic shape range.
        """
        for dynamic_hi in range(*fmap_range[0]):
            for dynamic_wi in range(*fmap_range[1]):
                dynamic_ho = (dynamic_hi + (pad_top + pad_bottom) - hk_dilation) // strideh + 1
                dynamic_wo = (dynamic_wi + (pad_left + pad_right) - wk_dilation) // stridew + 1
                if dynamic_ho < 1:
                    err_man.raise_err_specific_input_shape("conv2d",
                                                           "op [Conv2D] when h_in is {}, output " +
                                                           "featuremap h < 1, pleace check input range".format(
                                                               dynamic_hi))
                if dynamic_wo < 1:
                    err_man.raise_err_specific_input_shape("conv2d",
                                                           "op [Conv2D] when w_in is {}, output " +
                                                           "featuremap w < 1, pleace check input range".format(
                                                               dynamic_wi))
                if dynamic_ho == 1 or dynamic_wo == 1:
                    _check_load3d_constraint(dynamic_hi, dynamic_wi, dynamic_ho, dynamic_wo)
                if dynamic_wo > 1:
                    break
            if dynamic_ho > 1:
                break
    if dynamic_mode != "dynamic_hw":
        _check_load3d_constraint(h_i, w_i, h_out, w_out)
    else:
        _check_dynamic_range()

    w_block_size_n = CUBE_MKN[w_dtype]['mac'][2]
    shape_w[0] = ((shape_w[0] + w_block_size_n - 1) //
                  w_block_size_n)*w_block_size_n

    # filterH, filterW check(before dilation according to chip design demand )
    def _check_w_range():
        """
        Check width shape.
        """
        if shape_w[2] < FILTER_HW_MIN or shape_w[2] > FILTER_HW_MAX:
            range_value = "".join([str(FILTER_HW_MIN), ", ", str(FILTER_HW_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                 "kernel H", str(shape_w[2]))
        if shape_w[3] < FILTER_HW_MIN or shape_w[3] > FILTER_HW_MAX:
            range_value = "".join([str(FILTER_HW_MIN), ", ", str(FILTER_HW_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value,
                                                 "kernel W", str(shape_w[3]))

    def _check_stride():
        """
        Check stride.
        """
        if strideh < STRIDE_MIN or strideh > STRIDE_MAX:
            range_value = "".join([str(STRIDE_MIN), ", ", str(STRIDE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "strideh",
                                                 str(strideh))
        if stridew < STRIDE_MIN or stridew > STRIDE_MAX:
            range_value = "".join([str(STRIDE_MIN), ", ", str(STRIDE_MAX)])
            err_man.raise_err_attr_range_invalid("conv2d", range_value, "stridew",
                                                 str(stridew))
    _check_w_range()
    _check_pad()
    _check_stride()

    config = CUBE_MKN[w_dtype]
    ci0 = config['mac'][1]
    if ci0 <= 0:
        err_man.raise_err_specific("conv2d", "ci0 must > 0")

    fusion_para_check(fusion_para, pad_top, pad_bottom, shape_in)

    # check for not bigger than L1
    m_bit_ratio = {"float16": 2, "int8": 1}
    if dynamic_mode == "dynamic_hw":
        point_per_w = math.floor((fmap_range[1][1] - wk_dilation + pad_left +
                                  pad_right) / stridew) + 1
        w_in = math.floor(config['mac'][0] / point_per_w) + 2
        tmp = ((w_in - 1)*strideh + hk_dilation)*(fmap_range[1][1])
        max_feature_map_l1 = ci0*tmp*m_bit_ratio[w_dtype]
        _l1_buffer_size_check(max_feature_map_l1, fusion_para, "dynamci_hw")
    else:
        point_per_w = math.floor((w_i - wk_dilation + pad_left +
                                  pad_right) / stridew) + 1
        w_in = math.floor(config['mac'][0] / point_per_w) + 2
        tmp = ((int(w_in) - 1)*strideh + hk_dilation)*w_i
        max_feature_map_l1 = ci0*tmp*m_bit_ratio[w_dtype]
        if conv1d_split_w_flag:
            conv1d_filter_size = (shape_w[3] - 1)*wk_dilation + 1
            conv1d_min_l1 = (config['mac'][0] - 1)*stridew + conv1d_filter_size
            max_feature_map_l1 = ci0*conv1d_min_l1*m_bit_ratio[w_dtype]
        if not load2d_split_w_flag_set():
            _l1_buffer_size_check(max_feature_map_l1, fusion_para)

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
        cls.tiling_query_param.clear()
        cls.convbn1_flag = False
        cls.conv_deq_req_double_out = False
        cls.swrite_flag = False
        cls.swrite_dequant_flag = False
        cls.conv1d_split_w_flag = False

    tensor_map = {}
    dim_map = {}
    tiling = None
    tiling_query_param = {}
    convbn1_flag = False
    fusion_para = {"input_memory_type": 0, "output_memory_type": 0,
                   "valid_shape": (), "slice_offset": (),
                   "l1_fusion_type": -1,
                   "fmap_l1_addr_flag": False,
                   "fmap_l1_valid_size": -1}
    conv_deq_req_double_out = False
    swrite_flag = False
    swrite_dequant_flag = False
    conv1d_split_w_flag = False
    compress_index_shape = {}
    compress_tiling_ = {}
    compress_tiling_n = {}
    compress_tiling_n_frac = {}
    compress_tiling_frac = {}
    var_map = {}
    tiling_info_dict = {}
    fmap_range = None
    dynamic_mode = None
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
        elif isinstance(i, tvm.expr.Var):
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


def _fmap_c0_check_value(dtype, optim_dict):
    """
    This is fmap c0 check value.
    """
    fmap_c0_check_value = 4 if optim_dict["c0_optim_flg"] and optim_dict["use_v200_c04_flg"] and is_support_v200() \
        else CUBE_MKN[dtype]['mac'][1]

    return fmap_c0_check_value


OP_TAG = "convolution_"
TENSOR_MAP = {}
DIM_MAP = {}
NAME_INDEX = [0]


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
    weight = tvm.compute(weight_compress_shape, \
        lambda i, j, k, l: tvm.unzip(compress_index((j // \
            compress_tiling_n * compress_tiling_n_frac + \
            i // compress_tiling_k) * compress_tiling_frac * 8), \
        weight_compress(i, j, k, l)), \
        name='weight_unzip')
    res = conv(inputs, weight, para_dict, optim_dict, dsl_flag)
    return res


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
    def _v200_l0c2ub(c_col, res_dtype):
        """
        This is l0c to ub in v200 version.
        """
        c_ub = tvm.compute(ConvParam.mad_shape,
                           lambda *indices: c_col(*indices).astype(res_dtype),
                           name='C_UB',
                           tag=OP_TAG + "C_UB")
        TENSOR_MAP["c_ub"] = c_ub
        return c_ub

    def _fmap_ddr2l1(fmap, fmap_shape, strideh_opti_flag, valid_shape):
        """
        This is fmap from ddr to l1.
        """
        # remove invalid fmap data in h for optimization
        if strideh_opti_flag:
            if valid_shape:
                fmap_l1_shape = list(valid_shape)  # NC1HWC0
                fmap_l1_shape[2] = (fmap_l1_shape[2] - 1) // stride_h + 1

                offset = ConvParam.fusion_para["slice_offset"]
                _, _, h_offset, _, _ = offset

                fmap_l1 = tvm.compute(fmap_l1_shape, lambda n_idx, \
                    ci1_idx, \
                    hi_idx, \
                    wi_idx, \
                    ci0_idx: fmap[n_idx, ci1_idx, \
                    h_offset + hi_idx * stride_h, wi_idx, ci0_idx], \
                    name="fmap_l1")
                TENSOR_MAP["fmap_l1"] = fmap_l1
                return fmap_l1
            fmap_l1_shape = fmap_shape  # NC1HWC0
            fmap_l1_shape[2] = (fmap_l1_shape[2] - 1) // stride_h + 1

            fmap_l1 = tvm.compute(fmap_l1_shape, lambda n_idx, ci1_idx, \
                hi_idx, \
                wi_idx, \
                ci0_idx: fmap[n_idx, \
                ci1_idx, hi_idx*stride_h, wi_idx, ci0_idx], \
                name="fmap_l1")
            TENSOR_MAP["fmap_l1"] = fmap_l1
            return fmap_l1
        return None

    def _row_major_c0_value(fmap_shape, optim_dict):
        """
        Get the c0 value in row major.
        """
        # row major res c0 value
        row_major_c0_value = 4 if optim_dict["c0_optim_flg"] else fmap_shape[4]
        return row_major_c0_value

    def _v100_cal_im2col_row_major(fmap, fmap_im2col_row_major_shape,
                                   fmap_l1, optim):
        """
        Calculate im2col row major in v100 version.
        """
        filter_w, padding, stride, dilate, strideh_opti_flag = optim

        if strideh_opti_flag:
            fmap_im2col_row_major_res = im2col_row_major(
                fmap_im2col_row_major_shape, fmap_l1,
                filter_w, padding, (1, stride_w),
                dilate, fmap.dtype)
        else:
            fmap_im2col_row_major_res = im2col_row_major(
                fmap_im2col_row_major_shape,
                fmap, filter_w, padding, stride,
                dilate, fmap.dtype)
        TENSOR_MAP["fmap_im2col_row_major_res"] = fmap_im2col_row_major_res
        return fmap_im2col_row_major_res

    def _cube_compute(data, weight, mad_dtype, tiling=None,
                      optim_dict=None, bias=None):
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
            if valid_shape:
                fmap_shape = valid_shape

            batch_size = fmap_shape[0]
            feature_map_h = fmap_shape[2]
            feature_map_w = fmap_shape[3]

            if ConvParam.dynamic_mode == "dynamic_hw":
                height_out = ConvParam.var_map['ho']
                width_out = ConvParam.var_map['wo']
            else:
                height_out = ConvParam.h_out
                width_out = ConvParam.w_out

            return fmap_shape, height_out, width_out, batch_size, \
                feature_map_h, feature_map_w

        def _config_sread_flag():
            """
            Strided read flag set in tensor_map.
            """
            if fmap.op.tag == "strided_read":
                TENSOR_MAP["fmap"] = fmap.op.input_tensors[0]  # A_DDR
                TENSOR_MAP["strided_read_flag"] = True
                TENSOR_MAP["aipp_fuse_flag"] = False
            elif fmap.op.tag == "aipp_res_convolution":
                TENSOR_MAP["fmap"] = fmap.op.input_tensors[0]  # A_DDR
                TENSOR_MAP["aipp_fuse_flag"] = True
                TENSOR_MAP["strided_read_flag"] = False
            else:
                TENSOR_MAP["fmap"] = fmap
                TENSOR_MAP["strided_read_flag"] = False
                TENSOR_MAP["aipp_fuse_flag"] = False

        def _fusion_fmap_select(fmap):
            """
            Check L1 fusion fmap select.
            """
            valid_shape = ConvParam.fusion_para.get("valid_shape")

            offset = ConvParam.fusion_para.get("slice_offset")
            input_memory_type = ConvParam.fusion_para.get("input_memory_type")
            if offset and input_memory_type != 1:
                if TENSOR_MAP["strideh_opti_flag"] or \
                        TENSOR_MAP["l0a_load2d_flag"]:
                    # do it in _fmap_ddr2l1
                    pass
                else:
                    n_offset, c1_offset, h_offset, w_offset, c0_offset = offset
                    data_res = tvm.compute(valid_shape, lambda n, c1, h, w, c0:
                                           fmap(n + n_offset,
                                                c1 + c1_offset,
                                                h + h_offset,
                                                w + w_offset,
                                                c0 + c0_offset),
                                           name="fusion_fmap_select")
                    fmap = data_res
                    TENSOR_MAP['fusion_fmap_select'] = fmap
            return fmap

        def _mad_res(l0a_load2d_flag, valid_shape):
            """
            Calculate the mad result.
            """
            if l0a_load2d_flag:
                shape_al1_load2d = (batch_size,
                                    in_channel_c1,
                                    feature_map_h*feature_map_w,
                                    in_channel_c0)
                if valid_shape:
                    offset = ConvParam.fusion_para["slice_offset"]
                    if offset and len(offset) == 5:
                        _, _, h_offset, _, _ = offset
                else:
                    h_offset = 0
                al1_load2d = tvm.compute(shape_al1_load2d, \
                    lambda n, c1, m, c0: fmap(n, c1, \
                        (m // feature_map_w) + h_offset, \
                        m % feature_map_w, c0), \
                    name=OP_TAG + "al1_load2d")
                TENSOR_MAP["al1_load2d"] = al1_load2d

                shape_al0_load2d = (
                    batch_size,
                    int_ceil_div(feature_map_h*feature_map_w,
                                 CUBE_MKN[fmap.dtype]["mac"][0]),
                    in_channel_c1,
                    CUBE_MKN[fmap.dtype]["mac"][0],
                    in_channel_c0)

                al0_load2d = tvm.compute(shape_al0_load2d, \
                    lambda n, m_1, c1, m_0, c0: al1_load2d(n, c1, \
                        m_0 + CUBE_MKN[fmap.dtype]["mac"][0]*m_1, c0), \
                    name=OP_TAG + "al0_load2d")
                TENSOR_MAP["al0_load2d"] = al0_load2d

                c_col = mad(mad_shape, al0_load2d, weight, config, mad_dtype)
            else:
                c_col = mad(mad_shape, fmap_im2col_fractal_res, weight,
                            config, mad_dtype)
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

            l1_fusion_type = ConvParam.fusion_para.get("l1_fusion_type")
            if (l1_fusion_type == 0) or (l1_fusion_type == 1) or \
                    (input_memory_type == 1):
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
            if ConvParam.dynamic_mode is None:
                fmap_im2col_row_major_shape = (batch_size,
                                               height_out*width_out,
                                               in_channel_c1,
                                               filter_h,
                                               filter_w,
                                               in_channel_c0_row_major_res)
                fmap_im2col_row_major_res = \
                    _v100_cal_im2col_row_major(fmap,
                                               fmap_im2col_row_major_shape,
                                               fmap_l1,
                                               [filter_w, padding, stride,
                                                dilate, strideh_opti_flag])

                # im2col
                # small-z-big-Z
                howo_mad = (height_out*width_out + block_size_m -
                            1) // block_size_m*block_size_m
                k_size = \
                (in_channel_c0_row_major_res*in_channel_c1*filter_h*filter_w \
                    + block_size_k - 1) // block_size_k
                fmap_im2col_fractal_shape = (batch_size,
                                             howo_mad // block_size_m,
                                             k_size,
                                             block_size_m,
                                             block_size_k)
                fmap_im2col_fractal_res = im2col_fractal(
                    fmap_im2col_fractal_shape, fmap_im2col_row_major_res,
                    config, fmap.dtype)

                if is_support_v200() and not c04_v100_flag:
                    input_k_block = \
                    (in_channel_c1*filter_h*filter_w*in_channel_c0 + \
                        block_size_k - 1) // block_size_k * block_size_k
                    row_major_reshape_shape = \
                    (batch_size, howo_mad, input_k_block)
                    row_major_reshape_res = \
                        _im2col_row_major_reshape(row_major_reshape_shape,
                                                  fmap_im2col_row_major_res,
                                                  fmap.dtype)
                    fmap_im2col_fractal_res = \
                        _im2col_fractal_v200(fmap_im2col_fractal_shape,
                                             row_major_reshape_res,
                                             config)
                    TENSOR_MAP["row_major_reshape_res"] = row_major_reshape_res
                TENSOR_MAP["fmap_im2col_fractal_res"] = fmap_im2col_fractal_res
            else:
                howo_mad = (height_out * width_out + block_size_m - 1) \
                    // block_size_m * block_size_m
                fmap_im2col_fractal_shape = (batch_size, \
                    howo_mad // block_size_m, \
                    in_channel_c1 * filter_h * filter_w, \
                    block_size_m, \
                    block_size_k)
                if not strideh_opti_flag:
                    img2col_para = (fmap, filter_h, filter_w, padding, stride,
                                    width_out)
                else:
                    img2col_para = \
                    (fmap_l1, filter_h, filter_w, padding, (1, stride_w), \
                        width_out)
                fmap_im2col_fractal_res = img2col(
                    fmap_im2col_fractal_shape, img2col_para)
                TENSOR_MAP["fmap_im2col_fractal_res"] = fmap_im2col_fractal_res

            return howo_mad, fmap_im2col_fractal_res

        def _cal_bias_res():
            """
            Calculate bias result.
            """
            config = CUBE_MKN[w_dtype]
            if bias_optimize_flag:
                bias_ub_brc_shape = list(mad_shape)
                bias_ub_brc_shape[2] = bias_ub_brc_shape[2] // 16
                bias_ub_brc = tvm.compute(bias_ub_brc_shape, \
                    lambda i, j, k, l: bias_tensor(j * config['mac'][2] \
                        + l), \
                    name=OP_TAG + 'bias_ub_brc')
                bias_l0c = tvm.compute(mad_shape, lambda i1, j1, k_1, l1:
                                       bias_ub_brc(i1, j1, k_1 // 16, l1),
                                       name=OP_TAG + 'bias_l0c')
                TENSOR_MAP["bias_ub_brc"] = bias_ub_brc
                TENSOR_MAP["bias_l0c"] = bias_l0c
            else:
                bias_l0c = \
                    tvm.compute(mad_shape, lambda i1, j1, k_1, l1:
                                bias_tensor(j1 * config['mac'][2] + l1),
                                name=OP_TAG + 'bias_l0c')
                TENSOR_MAP["bias_l0c"] = bias_l0c

            TENSOR_MAP["bias_optimize_flag"] = bias_optimize_flag

            c_col = tvm.compute(mad_shape, lambda *index:
                                bias_l0c(*index) + TENSOR_MAP["c_col"](*index),
                                name=OP_TAG + 'c_col_bias')
            TENSOR_MAP["c_col_bias"] = c_col
            return bias_l0c, c_col

        fmap = data
        in_dtype = fmap.dtype

        _config_sread_flag()

        TENSOR_MAP["filter"] = weight
        strideh_opti_flag = (filter_h == 1 and stride_h > 1) \
            and not optim_dict["c0_optim_flg"] and sum(pad_h + pad_w) == 0

        if ConvParam.fusion_para.get("l1_fusion_type") == 1:
            # for L1  breadth fusion, fmap must load all at once
            strideh_opti_flag = False

        input_memory_type = ConvParam.fusion_para.get("input_memory_type")
        if input_memory_type == 1:
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

        fmap = _fusion_fmap_select(fmap)
        valid_shape = ConvParam.fusion_para.get("valid_shape")

        fmap_shape, height_out, width_out, batch_size, feature_map_h, \
            feature_map_w = _config_mmad_shape()
        config = CUBE_MKN[in_dtype]
        block_size_k = config['mac'][1]
        block_size_m = config['mac'][0]
        dilate = (dilate_h, dilate_w)

        # DDR -> L1
        fmap_l1 = _fmap_ddr2l1(fmap, fmap_shape, strideh_opti_flag,
                               valid_shape)

        # set_fmatrix
        # calculate im2col_row_major
        in_channel_c0_row_major_res = _row_major_c0_value(
            fmap_shape, optim_dict)
        howo_mad, fmap_im2col_fractal_res = \
        _cal_im2col_res(height_out, width_out)

        config = CUBE_MKN[res_dtype]
        mad_shape = (batch_size, (out_channel + config['mac'][2] - 1) // (
            config['mac'][2]), howo_mad, config['mac'][2])

        config = CUBE_MKN[w_dtype]
        ConvParam.mad_shape = mad_shape

        c_col = _mad_res(l0a_load2d_flag, valid_shape)

        TENSOR_MAP["c_col"] = c_col

        conv_shape = (batch_size, (out_channel + config['mac'][2] - 1) // (
            config['mac'][2]), height_out*width_out, config['mac'][2])
        DIM_MAP["out_img_shape"] = conv_shape
        DIM_MAP["out_img_height_width"] = [height_out, width_out]
        ConvParam.conv_shape = conv_shape
        filter_shape = [out_channel, filter_h, filter_w, 1]
        dim_map1 = im2col_dim(shape_to_list(fmap.shape),
                              filter_shape, list(padding), list(stride),
                              list(dilate), config)
        dim_map_copy = DIM_MAP.copy()
        dim_map_copy.update(dim_map1)
        ConvParam.dim_map = dim_map_copy
        ConvParam.tiling = tiling
        TENSOR_MAP["conv_vector_fused_flag"] = False
        TENSOR_MAP["bias_optimize_flag"] = False

        if isinstance(bias, tvm.tensor.Tensor):
            TENSOR_MAP["bias"] = bias
        bias_tensor_flag = isinstance(bias, tvm.tensor.Tensor)
        bias_optimize_flag = True
        if is_support_v200():
            bias_optimize_flag = False

        howo_mad = (height_out*width_out + block_size_m -
                    1) // block_size_m*block_size_m

        mad_shape = (batch_size, (out_channel + config['mac'][2] - 1) // (
            config['mac'][2]), howo_mad, config['mac'][2])
        conv_shape = DIM_MAP["out_img_shape"]
        if bias_tensor_flag:
            _, c_col = _cal_bias_res()

        ConvParam.tensor_map = TENSOR_MAP
        return c_col

    def conv_and_quant_compute(data, weight, mad_dtype, res_dtype, stride_h, \
        stride_w, dilate_h, dilate_w, filter_h, filter_w, bias=False, \
        no_vector=False, tiling=None, conv_fused_flag=False, \
        optim_dict=None, kernel_name=None):
        """
        conv

        Parameters
        ----------
        data : tvm.tensor, Feature Map

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
        fmap_shape = shape_to_list(fmap.shape)
        batch_size = fmap_shape[0]

        weight_shape = shape_to_list(weight.shape)
        out_channel_c1 = weight_shape[1]
        out_channel_c0 = weight_shape[2]
        out_channel = out_channel_c1*out_channel_c0

        if ConvParam.dynamic_mode == "dynamic_hw":
            height_out = ConvParam.var_map["ho"]
            width_out = ConvParam.var_map["wo"]
        else:
            height_out = ConvParam.h_out
            width_out = ConvParam.w_out

        config = CUBE_MKN[in_dtype]
        block_size_m = config['mac'][0]
        padding = ConvParam.padding
        stride = (stride_h, stride_w)
        dilate = (dilate_h, dilate_w)

        c_col = _cube_compute(fmap, weight, mad_dtype,
                              tiling, optim_dict, bias)

        howo_mad = (height_out*width_out + block_size_m -
                    1) // block_size_m*block_size_m

        mad_shape = (batch_size, \
            (out_channel + config['mac'][2] - 1) // (config['mac'][2]), \
            howo_mad, \
            config['mac'][2])
        config = CUBE_MKN[w_dtype]
        c_ub = tvm.compute(mad_shape, lambda n, i, j, k:
                           c_col(n, i, j, k).astype(res_dtype),
                           name='C_UB', tag=OP_TAG + "C_UB",
                           attrs={
                               'no_vector': no_vector,
                               'sqrt': False,
                               'res_dtype': res_dtype,
                               'kernel_h': filter_h,
                               'kernel_w': filter_w,
                               'padding': padding,
                               'stride': stride,
                               'dilate': dilate,
                               'width_out': width_out,
                               'kernel_name': kernel_name})

        filter_shape = [out_channel, filter_h, filter_w, 1]
        dim_map1 = im2col_dim(shape_to_list(fmap.shape), filter_shape,
                              list(padding), list(stride),
                              list(dilate), config)
        dim_map_copy = DIM_MAP.copy()
        dim_map_copy.update(dim_map1)

        TENSOR_MAP["c_ub"] = c_ub
        TENSOR_MAP["conv_vector_fused_flag"] = conv_fused_flag
        ConvParam.tensor_map = TENSOR_MAP
        ConvParam.dim_map = dim_map_copy
        ConvParam.tiling = tiling

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
            n_batch, col_h, col_w, block_size_h, block_size_w = idx

            virtual_h = col_h * block_size + block_size_h
            virtual_w = col_w * block_size + block_size_w

            back_c1 = virtual_w // block_size // kernel_w // kernel_h
            back_h = (virtual_h // fmap_wo) * stride[0] + \
            (col_w // kernel_w % kernel_h)
            back_w = (virtual_h % fmap_wo) * stride[1] + (col_w % kernel_w)

            return tvm.select(tvm.any(back_h < padding[0],
                                      back_h > fmap.shape[2] + padding[0] - 1,
                                      back_w < padding[2],
                                      back_w > fmap.shape[3] + padding[2] - 1),
                              tvm.const(0, fmap.dtype),
                              fmap(n_batch, back_c1, back_h - padding[0],
                                   back_w - padding[2], block_size_w))
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

    def im2col_dim(img_shape, filter_shape, pad, stride, dilate,
                   config):
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
        if ConvParam.dynamic_mode != "dynamic_hw":
            out_h = ((img_shape[-3] + pad[2] + pad[3]) -
                     ((filter_shape[-3]-1)*dilate[0] + 1)) // stride[0] + 1
            out_w = ((img_shape[-2] + pad[0] + pad[1]) -
                     ((filter_shape[-2]-1)*dilate[1] + 1)) // stride[1] + 1
        else:
            out_h = ConvParam.var_map['ho']
            out_w = ConvParam.var_map['wo']

        fmap_valid_dim = (batch, out_h*out_w, \
            img_shape[-4]*img_shape[-1]*filter_shape[-2]*filter_shape[-3])

        fmap_matrix_dim = (batch, \
            ((fmap_valid_dim[-2] + mac_dim[0] - 1) // mac_dim[0]), \
            ((fmap_valid_dim[-1] + mac_dim[1] - 1) // mac_dim[1]), \
            mac_dim[0], mac_dim[1])

        filter_valid_dim = (img_shape[-4]*filter_shape[-3]*filter_shape[-2]
                            * img_shape[-1], filter_shape[-4]*filter_shape[-1])

        filter_matrix_dim = ((filter_valid_dim[-2] + mac_dim[1] - 1) \
            // mac_dim[1], \
            (filter_valid_dim[-1] + mac_dim[2] - 1) // mac_dim[2], \
            mac_dim[2], mac_dim[1])

        return {
            "img_shape": img_shape,
            "fmap_matrix_dim": fmap_matrix_dim,
            "filter_matrix_dim": filter_matrix_dim}

    def im2col_row_major(
            fmap_im2col_vm_shape, fmap, kernel_w, padding, stride,
            dilate, compute_dtype):
        """
        calculate im2col_row_major tensor

        Parameters
        ----------
        fmap_im2col_vm_shape : shape of fmap_im2col_row_major

        fmap : feature map

        kernel_w: the kernel value in  w

        padding: the padding shape

        stride: the stride value

        dilate: the dilate value

        compute_dtype: dtype of compute result

        Returns
        -------
        fmap_im2col_row_major tensor
        """

        def __im2col_row_major_indices(batch, howo, cin_1, k_h, k_w, cin_0, \
            fmap, kernel_w, padding, stride, dilate):
            """
            calculate im2col_row_major tvm lambda function
            Parameters
            ----------
            indices : indices in lambda function

            fmap : feature map

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
            width_out = (input_w.value + padding_left + padding_right
                         - ((kernel_w - 1)*dilate_w + 1)) // (stride_w) + 1

            h_index = (howo // width_out)*stride_h + k_h*dilate_h
            w_index = (howo % width_out)*stride_w + k_w*dilate_w
            input_memory_type = ConvParam.fusion_para.get("input_memory_type")
            slice_offset = ConvParam.fusion_para.get("slice_offset")
            offset = slice_offset[2] if (slice_offset and
                                         input_memory_type == 1) else 0
            return tvm.select(
                tvm.any(h_index < padding_top,
                        h_index > input_h.value + padding_top - 1,
                        w_index < padding_left,
                        w_index > input_w.value + padding_left - 1),
                tvm.const(offset_x, compute_dtype),
                fmap(batch, cin_1, h_index - padding_top + offset,
                     w_index - padding_left, cin_0))

        return tvm.compute(fmap_im2col_vm_shape, lambda batch, howo, cin_1, \
            k_h, k_w, cin_0: __im2col_row_major_indices(batch, \
                howo, cin_1, k_h, k_w, cin_0, \
                fmap, kernel_w, padding, stride, dilate), \
            name='im2col_row_major', tag=OP_TAG + 'im2col_row_major')

    def im2col_fractal(fmap_im2col_shape, fmap,
                       config, compute_dtype):
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

        def __im2col_fractal_indices(batch, m_1, k_1, m_0, k_0, fmap):
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
            _, howo, _, kernel_h, kernel_w, _ = fmap.shape

            hw_index = m_1*block_size_m + m_0

            c1_index = (((k_1*block_size + k_0) // block_size) //
                        kernel_w.value) // kernel_h.value

            kh_index = (((k_1*block_size + k_0) // block_size) //
                        kernel_w.value) % kernel_h.value

            kw_index = ((k_1*block_size + k_0) // block_size) % kernel_w.value

            c0_index = (k_1*block_size + k_0) % block_size

            if optim_dict["c0_optim_flg"]:
                c1_index = 0
                kh_index = (k_1*4 + k_0 // 4) // kernel_w.value
                kw_index = (k_1*4 + k_0 // 4) % kernel_w.value
                c0_index = k_0 % 4
            dtype = compute_dtype

            return tvm.select(
                tvm.any(hw_index < 0, hw_index > howo.value - 1),
                tvm.const(0.0, dtype),
                fmap(batch, hw_index,
                     c1_index, kh_index, kw_index, c0_index))

        return tvm.compute(fmap_im2col_shape, lambda batch, m_1, k_1, m_0, k_0:
                           __im2col_fractal_indices(batch, m_1, k_1, m_0,
                                                    k_0, fmap),
                           name='im2col_fractal',
                           tag=OP_TAG + 'im2col_fractal')

    def _im2col_row_major_reshape(fmap_im2col_shape,
                                  fmap_row_major, compute_dtype):
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
        _, howo, input_c1, filter_h, filter_w, input_c0 = fmap_row_major.shape
        row_major_reshape = tvm.compute(
            fmap_im2col_shape, lambda i, j, k: tvm.select(
                tvm.all(k < input_c1*filter_h*filter_w*input_c0, j < howo),
                fmap_row_major(i, j, k // (filter_h*filter_w*input_c0),
                               k // (filter_w*input_c0) % filter_h,
                               k // (input_c0) % (filter_w),
                               k % input_c0), tvm.const(0.0, compute_dtype)),
            name="row_major_reshape",
            tag=OP_TAG + 'row_major_reshape')

        return row_major_reshape

    def _im2col_fractal_v200(fmap_im2col_shape,
                             im2col_row_major_reshape, config):
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
        res_im2col_fractal = tvm.compute(
            fmap_im2col_shape, lambda i, j, k, l, m: im2col_row_major_reshape(
                i, j*block_size_m + l, k*block_size_k + m),
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
        axis_k1 = tvm.reduce_axis((0, weight.shape[0]), name='cin_1_kh_kw')
        axis_k0 = tvm.reduce_axis((0, block_size), name='cin_0')

        if mad_dtype in ["float16", "int32"]:
            mode = 'f162f16'
        else:
            mode = 'f162f32'
        offset_d = offset_x if is_support_v200() else 0
        c_col = tvm.compute(
            mad_shape,
            lambda batch, cout_1, howo, cout_0:
            tvm.sum((
                (fmap[batch,
                      howo // block_size_m,
                      axis_k1,
                      howo % block_size_m,
                      axis_k0] - offset_d) *
                weight[axis_k1,
                       cout_1,
                       cout_0,
                       axis_k0]).astype(mad_dtype),
                    axis=[axis_k1, axis_k0]),
            name='mad1',
            tag=OP_TAG + "c_col",
            attrs={'mode': mode})
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
        dim_map = {}
        dim_map["out_img_shape"] = shape_to_list(in_tensor0.shape)
        NAME_INDEX[0] += 1

        with tvm.tag_scope('conv_vector_bias_add'):
            c_add_vector = \
                tvm.compute(dim_map["out_img_shape"], \
                    lambda *indice: in_tensor0(*indice) + \
                    in_tensor1(indice[1]*CUBE_MKN[in_tensor0.dtype]['mac'][2] \
                        + indice[3]), \
                    name='bias_add_vector' + "_cc_" + str(NAME_INDEX[0]), \
                    attrs={'width_out': in_tensor0.op.attrs["width_out"]})
        return c_add_vector

    def remove_pad(res, res_remove_pad_shape):
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
            res_tensor = tvm.compute(res_remove_pad_shape, \
                lambda *indice: res(*indice), \
                name='remove_pad' + "_cc_" + str(NAME_INDEX[0]))
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
        if isinstance(optim_dict, dict) and "c0_optim_flg" in \
                optim_dict.keys() and \
                isinstance(optim_dict["c0_optim_flg"], bool):
            pass
        else:
            err_man.raise_err_specific("conv2d", "Invalid optim_dict check")

        kernel_one_one = (para_dict["filter_h"] == 1) and \
            (para_dict["filter_w"] == 1)
        if optim_dict["c0_optim_flg"]:
            c0_value = _fmap_c0_check_value(weight.dtype, optim_dict)
            if weight.dtype != "int8" and data.shape[1].value == 1 \
                    and data.shape[4].value == c0_value \
                    and weight.shape[3].value == 16 and not kernel_one_one:
                pass
            else:
                err_man.raise_err_specific("conv2d", \
                    "Invalid config for c0=4 optimize feature.")

    def check_para_dict(para_dict, wtype):
        """
        Check conv params in para dict.
        """

        def check_para_dict_more():
            """
            Check more conv params in para dict.
            """
            if "mad_dtype" not in para_dict:
                if wtype == "int8":
                    mad_dtype = "int32"
                elif cce_conf.get_soc_spec("SOC_VERSION") in \
                        ("Hi3796CV300ES", "Hi3796CV300CS"):
                    mad_dtype = "float16"
                else:
                    mad_dtype = "float32"
                para_dict["mad_dtype"] = mad_dtype
            if "offset_x" not in para_dict:
                para_dict["offset_x"] = 0
            if "kernel_name" not in para_dict:
                para_dict["kernel_name"] = "conv2d"

        if not isinstance(para_dict, dict):
            err_man.raise_err_check_type("conv2d", "the third Input",
                                         "dict", "not dict")
        if "pad_h" not in para_dict:
            if "padh" in para_dict:
                para_dict["pad_h"] = para_dict["padh"]
            else:
                err_man.raise_err_specific_input_shape("conv2d",
                                                       "para_dict must contain pad_h")
        if "pad_w" not in para_dict:
            if "padw" in para_dict:
                para_dict["pad_w"] = para_dict["padw"]
            else:
                err_man.raise_err_specific_input_shape("conv2d", \
                    "para_dict must contain pad_w")
        if "stride_h" not in para_dict:
            if "strideh" in para_dict:
                para_dict["stride_h"] = para_dict["strideh"]
            else:
                err_man.raise_err_specific_input_shape("conv2d", \
                    "para_dict must contain stride_h")
        if "stride_w" not in para_dict:
            if "stridew" in para_dict:
                para_dict["stride_w"] = para_dict["stridew"]
            else:
                err_man.raise_err_specific_input_shape("conv2d", \
                    "para_dict must contain stride_w")
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
                err_man.raise_err_specific_input_shape("conv2d", \
                    "para_dict must contain filter_h")
        if "filter_w" not in para_dict:
            if "filterw" in para_dict:
                para_dict["filter_w"] = para_dict["filterw"]
            else:
                err_man.raise_err_specific_input_shape("conv2d", \
                    "para_dict must contain filter_w")

        check_para_dict_more()

    def check_data(data, optim_dict):
        """
        Check conv fmap param.
        """
        if not isinstance(data, tvm.tensor.Tensor):
            err_man.raise_err_specific("conv2d", \
                "the first Input parameter must be a tvm.tensor.Tensor")
        if len(data.shape) != 5:
            err_man.raise_err_specific("conv2d", \
                "the first Input parameter must be a 5 dim tvm.tensor.Tensor")
        check_dtype_list = ('float16', )
        if is_support_v200() or cce_conf.get_soc_spec("SOC_VERSION") in \
                ("Ascend310", "Hi3796CV300ES"):
            check_dtype_list = ('int8', "float16")
        util.check_dtype_rule(data.dtype, check_dtype_list)

        block_size_k = 4 \
            if optim_dict["c0_optim_flg"] and is_support_v200() and optim_dict["use_v200_c04_flg"] \
            else CUBE_MKN[data.dtype]['mac'][1]
        if data.shape[4].value != block_size_k:
            err_man.raise_err_scene_equal_limitation("conv2d", \
                "the last dim of first Input parameter", str(block_size_k))

    def check_weight(weight):
        """
        Check conv weight param.
        """
        if not isinstance(weight, tvm.tensor.Tensor):
            err_man.raise_err_specific("conv2d", \
                "the first Input parameter must be a tvm.tensor.Tensor")
        if len(weight.shape) != 4:
            err_man.raise_err_specific("conv2d", \
                "the first Input parameter must be a 4 dim tvm.tensor.Tensor")

        check_dtype_list = ('float16', )
        if is_support_v200() or cce_conf.get_soc_spec("SOC_VERSION") in \
                ("Ascend310", "Hi3796CV300ES"):
            check_dtype_list = ('int8', "float16")

        util.check_dtype_rule(weight.dtype, check_dtype_list)
        block_size_k = CUBE_MKN[weight.dtype]['mac'][1]

        if weight.shape[3].value != block_size_k:
            err_man.raise_err_scene_equal_limitation("conv2d", \
                "the last dim of first Input parameter", str(block_size_k))

    def _save_conv_param():
        """
        save conv params in ConvParam.
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
        ConvParam.fusion_para = para_dict.get("fusion_para")
        ConvParam.kernel_name = para_dict.get("kernel_name")
        ConvParam.batch = shape_in[0]

        ConvParam.h_in = shape_in[2]
        if para_dict.get("fusion_para").get("valid_shape"):
            ConvParam.h_in = para_dict.get("fusion_para").get("valid_shape")[2]
        ConvParam.w_in = shape_in[3]

        ConvParam.padding = [ConvParam.pad_h[0], ConvParam.pad_h[1],
                             ConvParam.pad_w[0], ConvParam.pad_w[1]]

        filter_h_dilation = (ConvParam.filter_h - 1)*ConvParam.dilate_h + 1
        filter_w_dilation = (ConvParam.filter_w - 1)*ConvParam.dilate_w + 1
        ConvParam.h_out = (ConvParam.h_in + (ConvParam.pad_h[0] + \
            ConvParam.pad_h[1]) - filter_h_dilation) // ConvParam.stride_h + 1
        ConvParam.w_out = (ConvParam.w_in + (ConvParam.pad_w[0] + \
            ConvParam.pad_w[1]) - filter_w_dilation) // ConvParam.stride_w + 1

    def _get_fmap_shape_nc1hwc0():
        """
        Get the 5HD format fmap shape.
        """
        if para_dict.get("dynamic_mode"):
            fmap_shape_nc1hwc0 = tuple(list(data.shape))
        elif para_dict.get("fusion_para"):
            ConvParam.fusion_para = para_dict["fusion_para"]
            valid_shape = ConvParam.fusion_para.get("valid_shape")
            if valid_shape:
                fmap_shape_nc1hwc0 = ConvParam.fusion_para.get("valid_shape")
            else:
                fmap_shape_nc1hwc0 = tuple(shape_to_list(data.shape))
        else:
            fmap_shape_nc1hwc0 = tuple(shape_to_list(data.shape))
        return list(fmap_shape_nc1hwc0)

    def _get_dsl_fmap_shape_nc1hwc0():
        """
        Get fmap_shape_nc1hwc0 for dsl interface.
        """
        valid_shape = ConvParam.fusion_para.get("valid_shape")
        if valid_shape:
            if ConvParam.dynamic_mode is None:
                fmap_shape_nc1hwc0 = tuple(shape_to_list(valid_shape))
            else:
                fmap_shape_nc1hwc0 = tuple(valid_shape)
        else:
            if ConvParam.dynamic_mode is None:
                fmap_shape_nc1hwc0 = tuple(shape_to_list(data.shape))
            else:
                fmap_shape_nc1hwc0 = tuple(data.shape)
        return fmap_shape_nc1hwc0

    def _fusion_para_get():
        """
        Get fusion paras.
        """
        if para_dict.get("fusion_para") is None:
            para_dict["fusion_para"] = {"input_memory_type": 0,
                                        "output_memory_type": 0,
                                        "valid_shape": (),
                                        "slice_offset": (),
                                        "l1_fusion_type": -1,
                                        "fmap_l1_addr_flag": False,
                                        "fmap_l1_valid_size": -1}

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
        return load2d_to_load3d_flag

    def _conv1d_spilit_w_flag_set():
        """
        Set this flag to define whether is doing conv1d.
        """
        if ConvParam.dynamic_mode is None and data.shape[2].value == 1 \
        and para_dict.get("filter_h") == 1 \
        and (pad_top + pad_bottom) == 0 and \
        not load2d_to_load3d_flag:
            ConvParam.conv1d_split_w_flag = True
        else:
            ConvParam.conv1d_split_w_flag = False

    def _save_tiling_info_dict(shape_fmap_nc1hwc0, shape_w_nc1hwc0,
                               c_ub_shape, in_dtype, w_dtype, res_dtype,
                               bias_flag, kernel_name):
        """
        Save tiling_info_dict for dynamic.
        """
        if ConvParam.dynamic_mode:
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
            w_out = (in_size_w + (pad_w[0] + pad_w[1] - wk_dilation)) // \
                stride_w + 1
            h_out = (in_size_h + (pad_h[0] + pad_h[1] - hk_dilation)) // \
                stride_h + 1
            c_shape = [c_ub_shape[0], c_ub_shape[1],
                       h_out, w_out, c_ub_shape[3]]
            ConvParam.tiling_info_dict = {
                "op_type": 'conv2d',
                "a_shape": list(shape_fmap_nc1hwc0),
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
                "fused_coefficient":
                [0, 0, 0],
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
        if ConvParam.dynamic_mode == "dynamic_batch":
            fmap_range = [get_te_var("batch_n").get_bound()]
        elif ConvParam.dynamic_mode == "dynamic_hw":
            fmap_range = [get_te_var("fmap_h").get_bound(),
                          get_te_var("fmap_w").get_bound()]
        else:
            return None
        dynamic_para = {
            "dynamic_mode": ConvParam.dynamic_mode,
            "fmap_range": fmap_range
        }
        return dynamic_para

    def _get_dynamic_mode():
        """
        Return dynamic mode.
        """
        if isinstance(data.shape[0], tvm.expr.Var):
            return "dynamic_batch"
        if isinstance(data.shape[2], tvm.expr.Var) and \
                isinstance(data.shape[3], tvm.expr.Var):
            return "dynamic_hw"
        return None

    def _get_var_map():
        """
        Get dynamic mode for ConvParam.
        """
        if ConvParam.dynamic_mode == "dynamic_batch":
            return {"batch_n": get_te_var("batch_n").get_tvm_var()}
        if ConvParam.dynamic_mode == "dynamic_hw":
            return {v: get_te_var(v).get_tvm_var()
                    for v in ("fmap_h", "fmap_w", "ho", "wo")}
        return None

    ConvParam.set_default()
    ConvParam.kernel_name = para_dict.get("kernel_name")
    ConvParam.dynamic_mode = _get_dynamic_mode()
    ConvParam.var_map = _get_var_map()

    if optim_dict is None:
        optim_dict = {"c0_optim_flg": False, "use_v200_c04_flg": False}
    pad_h = para_dict["pad_h"]
    pad_w = para_dict["pad_w"]
    if isinstance(pad_h, int):
        para_dict["pad_h"] = [pad_h, pad_h]
    pad_top = para_dict["pad_h"][0]
    pad_bottom = para_dict["pad_h"][1]
    if isinstance(pad_w, int):
        para_dict["pad_w"] = [pad_w, pad_w]
    pad_left = para_dict["pad_w"][0]
    pad_right = para_dict["pad_w"][1]

    ConvParam.conv1d_split_w_flag = False  # set conv1d off
    load2d_to_load3d_flag = load2d_to_load3d_flag_set()
    _conv1d_spilit_w_flag_set()

    check_optim_dict(optim_dict, para_dict, data, weight)
    check_data(data, optim_dict)
    check_weight(weight)
    check_para_dict(para_dict, weight.dtype)
    kernel_name = para_dict["kernel_name"]

    in_dtype = data.dtype
    w_dtype = weight.dtype

    stride_h = para_dict["stride_h"]
    stride_w = para_dict["stride_w"]
    dilate_h = para_dict["dilate_h"]
    dilate_w = para_dict["dilate_w"]
    filter_h = para_dict["filter_h"]
    filter_w = para_dict["filter_w"]
    offset_x = para_dict["offset_x"]
    mad_dtype = para_dict.get("mad_dtype")

    data_shape = shape_to_list(data.shape)
    in_channel_c1 = data_shape[1]
    in_channel_c0 = data_shape[4]
    in_channel_c = in_channel_c1*in_channel_c0
    if optim_dict["c0_optim_flg"]:
        in_channel_c = 4
    weight_shape = shape_to_list(weight.shape)
    out_channel_c1 = weight_shape[1]
    out_channel_c0 = weight_shape[2]
    out_channel = out_channel_c1*out_channel_c0

    shape_in = [data_shape[0], int(in_channel_c),
                data_shape[2], data_shape[3]]
    shape_w = [out_channel, int(in_channel_c), filter_h, filter_w]

    _fusion_para_get()
    _save_conv_param()

    if not optim_dict["c0_optim_flg"] and \
            int(data_shape[1]) != int((weight_shape[0] / filter_h) / filter_w):
        err_man.raise_err_scene_equal_limitation("conv2d", \
            "data_shape[1]", "((weight_shape[0]/filter_h)/filter_w)")

    res_dtype = "float16"
    if (is_support_v200() or cce_conf.get_soc_spec("SOC_VERSION") in \
        ("Ascend310", "Hi3796CV300ES")) and \
    (in_dtype, w_dtype) == ("int8", "int8"):
        res_dtype = "int32"
    ConvParam.res_dtype = res_dtype
    dynamic_para = _get_dynamic_para()
    check_conv_shape(
        shape_in, shape_w, pad_top, pad_bottom, pad_left,
        pad_right, stride_h, stride_w, in_dtype, w_dtype,
        para_dict["fusion_para"],
        optim_dict, dilateh=dilate_h, dilatew=dilate_w,
        dynamic_para=dynamic_para)

    block_size_k = CUBE_MKN[w_dtype]['mac'][1]
    if optim_dict["c0_optim_flg"]:
        block_size_k = 4
        in_channel_c = 4
    shape_w_nc1hwc0 = (
        out_channel, (in_channel_c + block_size_k - 1) //
        block_size_k, filter_h, filter_w, block_size_k)

    bias_tensor = None
    bias_tensor_flag = False

    if "bias_tensor" in para_dict.keys():
        bias_tensor = para_dict["bias_tensor"]
        if isinstance(bias_tensor, tvm.tensor.Tensor):
            bias_tensor_flag = True

    shape_fmap_nc1hwc0 = _get_dsl_fmap_shape_nc1hwc0()
    # for tiling c0 optim
    shape_fmap_nc1hwc0 = list(shape_fmap_nc1hwc0)
    shape_fmap_nc1hwc0[4] = _row_major_c0_value(
        shape_fmap_nc1hwc0, optim_dict)
    tiling = None

    default_tiling = False
    if ("default_tiling" in ConvParam.tiling_query_param) and \
            ConvParam.tiling_query_param["default_tiling"]:
        default_tiling = ConvParam.tiling_query_param["default_tiling"]

    ConvParam.tiling_query_param = {
        "fmap_shape_nc1hwc0": shape_fmap_nc1hwc0,
        "shape_w_nc1hwc0": shape_w_nc1hwc0,
        "in_dtype": in_dtype,
        "w_dtype": w_dtype,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "padw": pad_w,
        "padh": pad_h,
        "strideh": stride_h,
        "stridew": stride_w,
        "dilateh": dilate_h,
        "dilatew": dilate_w,
        "bias_flag": bias_tensor_flag,
        "default_tiling": default_tiling}

    if (is_support_v200() or cce_conf.get_soc_spec("SOC_VERSION") in
            ("Ascend310", "Hi3796CV300ES")) and in_dtype == "int8":
        if dsl_flag:
            ConvParam.tiling_query_param["bias_flag"] = bias_tensor_flag
            conv_res = _cube_compute(data, weight, mad_dtype, \
                tiling=tiling, optim_dict=optim_dict, bias=bias_tensor)
            res_remove_pad_shape = list(conv_res.shape)
            res_remove_pad_shape[2] = DIM_MAP["out_img_shape"][2]
            res_remove_pad = remove_pad(conv_res, res_remove_pad_shape)
            TENSOR_MAP["l0c_remove_pad"] = res_remove_pad
            return res_remove_pad
        conv_res = _cube_compute(data, weight, mad_dtype, tiling=tiling,
                                 optim_dict=optim_dict, bias=bias_tensor)
        res = _v200_l0c2ub(conv_res, res_dtype)
    else:
        no_vector_flag = False
        if (not dsl_flag) and (not bias_tensor_flag):
            no_vector_flag = True

        conv_res = conv_and_quant_compute(
            data, weight, mad_dtype, res_dtype,
            stride_h, stride_w, dilate_h, dilate_w,
            filter_h, filter_w, bias=False, no_vector=no_vector_flag,
            tiling=tiling, conv_fused_flag=dsl_flag,
            optim_dict=optim_dict, kernel_name=kernel_name)
        res = conv_res

        if bias_tensor_flag:
            TENSOR_MAP["fp16_bias"] = bias_tensor
            fp16_bias_res = bias_add(conv_res, bias_tensor)
            res = fp16_bias_res

    _save_tiling_info_dict(shape_fmap_nc1hwc0, shape_w_nc1hwc0,
                           list(res.shape), in_dtype, w_dtype,
                           res_dtype, bias_tensor_flag, kernel_name)

    if dsl_flag:
        res_c = tvm.compute(ConvParam.dim_map["out_img_shape"],
                            lambda batch, cout1, howo, cout0:
                            res(batch, cout1, howo, cout0),
                            name='C',
                            tag=OP_TAG + "C",
                            attrs={"width_out": ConvParam.w_out})
        ConvParam.tensor_map["C"] = res_c
        return res_c
    res_remove_pad_shape = list(res.shape)
    res_remove_pad_shape[2] = DIM_MAP["out_img_shape"][2]
    res_remove_pad = remove_pad(res, res_remove_pad_shape)
    return res_remove_pad
