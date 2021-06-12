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
trans_data
"""
from te.utils import para_check
from impl import four_2_five
from impl import five_2_four
from impl import zn_2_nchw
from impl import nchw_hwcn_zn
from impl import zn_2_hwcn
from impl import depthwise_weight_4d_2_6d
from impl import depthwise_weight_6d_2_4d
from impl import trans_data_2d
from impl import nz_2_nd
from impl import nd_2_nz
from impl import four_2_five_int8
from impl import five_2_four_int8
from impl import transpose_d
from impl import nd_2_zn_int8
from impl import ndhwc_2_ndc1hwc0
from impl import ndc1hwc0_2_ndhwc
from impl import nhwc_2_fractal_z_c04
from impl import nchw_2_fractal_z_c04
from impl import hwcn_2_fractal_z_c04
from impl import four_2_five_c04
from impl import dhwcn_2_fractal_z_3d
from impl import fractal_z_3d_2_dhwcn
from impl import nc1hwc0_2_nz
from impl import fractal_nz_2_nc1hwc0
from impl import zn_2_hwcn_lstm
from impl import ncdhw_2_ndc1hwc0
from impl import ndc1hwc0_2_ncdhw
from impl import ncdhw_2_fractal_z_3d
from impl import fractal_z_3d_2_ncdhw
from impl import ndhwc_2_fractal_z_3d
from impl import fractal_z_3d_2_ndhwc
from impl import zng_2_nchw_hwcn
from impl import nchw_2_fractal_z_g
from impl import hwcn_2_fractal_z_g
from impl import trans_data_positive_source_ntc
from impl import trans_data_negative_target_ntc
from impl import trans_data_positive_source_tc
from tbe.dsl.compute import cube_util
from tbe.tvm import api as tvm

# pylint: disable=locally-disabled,redefined-builtin,too-many-statements
# pylint: disable=too-many-arguments
def check_whether_2d(format, input_dict):
    """Check whether the 4D is 2D extend to 4D

    Parameters
    ----------
    format: str
        source data format, can be NHWC, NCHW, FRACTAL_Zn etc.
    input_dict: dict
        shape and dtype of output, should be same shape and type as input

    Returns
    -------
    is_2d : bool
        is_2d
    """
    is_2d = False
    shape = input_dict.get("shape")
    if not (len(list(format)) == len(shape) and len(shape) == 4):
        return is_2d

    dict_zip = dict(zip(list(format), shape))
    if dict_zip["H"] == 1 and dict_zip["W"] == 1 and \
            dict_zip["C"] % 16 == 0:
        is_2d = True

    return is_2d


# pylint: disable=locally-disabled,too-many-branches
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_STR,
                            para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def trans_data(src, dst, src_format, dst_format, groups=1,
               kernel_name='trans_data'):
    """
    algorithm: format_transfer
    doing format_transfer for various data format
    only support NHWC/NCHW to NC1HWC0 and NC1HWC0 to NHWC/NCHW
    NCHW to FRACTAL_Zn or FRACTAL_Zn to NCHW
    HWCN to FRACTAL_Zn or FRACTAL_Zn to HWCN

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    src_format: str
        source data format, can be NHWC, NCHW, FRACTAL_Zn etc.
    dst_format: str
        target data format, can be NC1HWC0, NCHW, FRACTAL_Zn etc.
    groups: int
        default 1
    kernel_name: str
        kernel name, default value is "format_transfer"

    Returns
    -------
    None
    """
    if src_format.upper() == "NC1HWC0" and dst_format.upper() == "NCHW" and \
            src.get("dtype") == "int8" and src.get("shape")[-1] == 32:
        trans_data_negative_target_ntc.trans_data_negative_target_ntc(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "NHWC" and dst_format.upper() == "NC1HWC0" and \
            src.get("dtype") == "int8" and dst.get("shape")[-1] == 32:
        trans_data_positive_source_tc.trans_data_positive_source_tc(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "NCHW" and dst_format.upper() == "NC1HWC0" and \
            src.get("dtype") == "int8" and dst.get("shape")[-1] == 32:
        trans_data_positive_source_ntc.trans_data_positive_source_ntc(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "HWCN" and dst_format.upper() == "FRACTAL_Z" and \
            src.get("dtype") == "int8" and dst.get("shape")[-1] == 32 and groups == 1:
        trans_data_positive_source_ntc.trans_data_positive_source_ntc(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "FRACTAL_Z" and dst_format.upper() == "NCHW" and \
            src.get("dtype") == "int8" and src.get("shape")[-1] == 32 and groups == 1:
        trans_data_negative_target_ntc.trans_data_negative_target_ntc(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper() == "NHWC" or src_format.upper() == "NCHW") \
            and dst_format.upper() == "NC1HWC0":
        if check_whether_2d(src_format.upper(), src):
            trans_data_2d(src, dst, src_format, dst_format, kernel_name)
        else:
            if src.get("dtype") == "int8" or src.get("dtype") == "bool" \
                    or src.get("dtype") == "uint8":
                four_2_five_int8.four_2_five(src, dst, src_format,
                                             dst_format, kernel_name)
            else:
                four_2_five.four_2_five(src, dst, src_format,
                                        dst_format, kernel_name)
    elif src_format.upper() == "NC1HWC0" \
            and (dst_format.upper() == "NHWC" or dst_format.upper() == "NCHW"):
        if check_whether_2d(dst_format.upper(), dst):
            trans_data_2d(src, dst, src_format, dst_format, kernel_name)
        else:
            if src.get("dtype") == "int8" or src.get("dtype") == "bool" \
                    or src.get("dtype") == "uint8":
                five_2_four_int8.five_2_four(src, dst, src_format,
                                             dst_format, kernel_name)
            else:
                five_2_four.five_2_four(src, dst, src_format,
                                        dst_format, kernel_name)
    elif src_format.upper() == "NCHW" \
            and ((dst_format.upper() == "FRACTAL_ZN" or dst_format.upper() == "FRACTAL_Z") and groups == 1):
        trans_data_positive_source_ntc.trans_data_positive_source_ntc(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "NCHW" \
            and ((dst_format.upper() == "FRACTAL_ZN" or dst_format.upper() == "FRACTAL_Z") and groups > 1):
        nchw_2_fractal_z_g.nchw_2_fractal_z_g(src, dst, src_format, dst_format, groups, kernel_name)
    elif src_format.upper() == "ND" \
            and (dst_format.upper() == "FRACTAL_ZN"
                 or dst_format.upper() == "FRACTAL_Z"):
        nd_2_zn_int8.nd_2_zn_int8(src, dst, src_format,
                                  dst_format, kernel_name)
    elif (src_format.upper() == "FRACTAL_ZN"
          or src_format.upper() == "FRACTAL_Z") \
            and dst_format.upper() == "NCHW" and groups == 1:
        zn_2_nchw.zn_2_nchw(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper() == "FRACTAL_ZN"
          or src_format.upper() == "FRACTAL_Z") \
            and dst_format.upper() == "NCHW" and groups > 1:
        zng_2_nchw_hwcn.zng_2_nchw_hwcn(src, dst, src_format, dst_format, groups, kernel_name)
    elif src_format.upper() == "FRACTAL_Z_3D" and dst_format.upper() == "DHWCN" and groups > 1:
        zng_2_nchw_hwcn.zng_2_nchw_hwcn(src, dst, src_format, dst_format, groups, kernel_name)
    elif src_format.upper() == "HWCN" \
            and ((dst_format.upper() == "FRACTAL_ZN"
                 or dst_format.upper() == "FRACTAL_Z"
                 or dst_format.upper() == "FRACTAL_ZN_LSTM") and groups == 1):
        nchw_hwcn_zn.nchw_hwcn_zn(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "HWCN" \
            and ((dst_format.upper() == "FRACTAL_ZN"
                 or dst_format.upper() == "FRACTAL_Z") and groups > 1):
        hwcn_2_fractal_z_g.hwcn_2_fractal_z_g(src, dst, src_format, dst_format, groups, kernel_name)
    elif src_format.upper() == "FRACTAL_ZN_LSTM" and \
            dst_format.upper() == "HWCN":
        zn_2_hwcn_lstm.zn_2_hwcn_lstm(src, dst, src_format,
                                      dst_format, kernel_name)
    elif (src_format.upper() == "FRACTAL_ZN"
          or src_format.upper() == "FRACTAL_Z") \
            and dst_format.upper() == "HWCN" and groups == 1:
        zn_2_hwcn.zn_2_hwcn(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper() == "FRACTAL_ZN"
          or src_format.upper() == "FRACTAL_Z") \
            and dst_format.upper() == "HWCN" and groups > 1:
        zng_2_nchw_hwcn.zng_2_nchw_hwcn(src, dst, src_format, dst_format, groups, kernel_name)
    elif src_format.upper() == "HWCN" \
            and dst_format.upper() == "C1HWNCOC0":
        depthwise_weight_4d_2_6d(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "C1HWNCOC0" \
            and dst_format.upper() == "HWCN":
        depthwise_weight_6d_2_4d(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper() == "NHWC" or src_format.upper() == "NCHW"\
          or src_format.upper() == "ND") and \
            dst_format.upper() == "FRACTAL_NZ":
        nd_2_nz.nd_2_nz(src, dst, src_format, dst_format, kernel_name)
    elif (src_format.upper() == "FRACTAL_NZ" or
          src_format == "FORMAT_FRACTAL_Nz") and \
            (dst_format in ("ND", "NHWC", "NCHW")):
        nz_2_nd.nz_2_nd(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "NCHW" and dst_format.upper() == "NHWC":
        transpose_d(src, dst, [0, 2, 3, 1], kernel_name)
    elif src_format.upper() == "NCHW" and dst_format.upper() == "HWCN":
        transpose_d(src, dst, [2, 3, 1, 0], kernel_name)
    elif src_format.upper() == "NHWC" and dst_format.upper() == "NCHW":
        transpose_d(src, dst, [0, 3, 1, 2], kernel_name)
    elif src_format.upper() == "NHWC" and dst_format.upper() == "HWCN":
        transpose_d(src, dst, [1, 2, 3, 0], kernel_name)
    elif src_format.upper() == "HWCN" and dst_format.upper() == "NCHW":
        transpose_d(src, dst, [3, 2, 0, 1], kernel_name)
    elif src_format.upper() == "HWCN" and dst_format.upper() == "NHWC":
        transpose_d(src, dst, [3, 0, 1, 2], kernel_name)
    elif src_format.upper() == "CHWN" and dst_format.upper() == "NCHW":
        transpose_d(src, dst, [3, 0, 1, 2], kernel_name)
    elif src_format.upper() == "CHWN" and dst_format.upper() == "NHWC":
        transpose_d(src, dst, [3, 1, 2, 0], kernel_name)
    elif src_format.upper() == "CHWN" and dst_format.upper() == "HWCN":
        transpose_d(src, dst, [1, 2, 0, 3], kernel_name)
    elif src_format.upper() == "NDHWC" and dst_format.upper() == "NDC1HWC0":
        ndhwc_2_ndc1hwc0.ndhwc_2_ndc1hwc0(src, dst, src_format,
                                          dst_format, kernel_name)
    elif src_format.upper() == "NDC1HWC0" and dst_format.upper() == "NDHWC":
        ndc1hwc0_2_ndhwc.ndc1hwc0_2_ndhwc(src, dst, src_format,
                                          dst_format, kernel_name)
    elif src_format.upper() == "NHWC" and \
            dst_format.upper() == "FRACTAL_Z_C04":
        nhwc_2_fractal_z_c04.nhwc_2_fractal_z_c04(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif src_format.upper() == "NCHW" and \
            dst_format.upper() == "FRACTAL_Z_C04":
        nchw_2_fractal_z_c04.nchw_2_fractal_z_c04(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif src_format.upper() == "HWCN" and \
            dst_format.upper() == "FRACTAL_Z_C04":
        hwcn_2_fractal_z_c04.hwcn_2_fractal_z_c04(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif (src_format.upper() in ["NHWC", "NCHW", "HWCN"]) and \
            dst_format.upper() == "NC1HWC0_C04":
        four_2_five_c04.four_2_five_c04(src, dst, src_format,
                                        dst_format, kernel_name)
    elif src_format.upper() == "DHWCN" and \
            dst_format.upper() == "FRACTAL_Z_3D":
        dhwcn_2_fractal_z_3d.dhwcn_2_fractal_z_3d(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif src_format.upper() == "FRACTAL_Z_3D"\
            and dst_format.upper() == "DHWCN" and groups == 1:
        fractal_z_3d_2_dhwcn.fractal_z_3d_2_dhwcn(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif src_format.upper() == "NC1HWC0" and \
            dst_format.upper() == "FRACTAL_Z":
        nc1hwc0_2_nz.nc1hwc0_2_nz(src, dst, src_format,
                                  dst_format, kernel_name)
    elif src_format.upper() == "FRACTAL_NZ"\
            and dst_format.upper() == "NC1HWC0":
        fractal_nz_2_nc1hwc0.fractal_nz_2_nc1hwc0(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif src_format.upper() == "NCDHW" and dst_format.upper() == "NDC1HWC0":
        ncdhw_2_ndc1hwc0.ncdhw_2_ndc1hwc0(src, dst, src_format,
                                          dst_format, kernel_name)
    elif src_format.upper() == "NDC1HWC0" and dst_format.upper() == "NCDHW":
        ndc1hwc0_2_ncdhw.ndc1hwc0_2_ncdhw(src, dst, src_format,
                                          dst_format, kernel_name)
    elif src_format.upper() == "NCDHW" and dst_format.upper() == "FRACTAL_Z_3D":
        ncdhw_2_fractal_z_3d.ncdhw_2_fractal_z_3d(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif src_format.upper() == "FRACTAL_Z_3D" and dst_format.upper() == "NCDHW":
        fractal_z_3d_2_ncdhw.fractal_z_3d_2_ncdhw(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif src_format.upper() == "NDHWC" and dst_format.upper() == "FRACTAL_Z_3D":
        ndhwc_2_fractal_z_3d.ndhwc_2_fractal_z_3d(src, dst, src_format,
                                                  dst_format, kernel_name)
    elif src_format.upper() == "FRACTAL_Z_3D" and dst_format.upper() == "NDHWC":
        fractal_z_3d_2_ndhwc.fractal_z_3d_2_ndhwc(src, dst, src_format,
                                                  dst_format, kernel_name)
    else:
        raise RuntimeError("not support this kind of format transfer !")


def trans_data_compute(src, dst, src_format, dst_format, groups=1, kernel_name='transdata'):
    """
    algorithm: format_transfer
    used for on the fly format transformation , For example NHWC TO NC1HWC0,
    NC1HWC0 TO NHWC, NHWC TO FRACTAL_Z
    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst: dict
        shape and dtype of output, should be same shape and type as input
    src_format: str
        source data format, can be NHWC, NCHW, FRACTAL_Zn etc.
    dst_format: str
        target data format, can be NC1HWC0, NCHW, FRACTAL_Zn etc.
    groups: int
        default 1
    kernel_name: str
        kernel name, default value is "format_transfer"

    Returns
    -------
    None
    """
    dst_tensor = None
    c0_dict = {"float32": 8, "float16": 16, "int8": 32, "int4": 64}

    if src_format == "NHWC" and dst_format == "NC1HWC0":
        src_n, src_h, src_w, src_c = tuple(i.value for i in src.shape)
        dst_c0 = c0_dict.get(src.dtype)
        dst_c1 = cube_util.ceil_div(src_c, dst_c0)
        dst_shape = (src_n, dst_c1, src_h, src_w, dst_c0)
        dst_tensor = tvm.compute(dst_shape,
            lambda n_idx, c1_idx, h_idx, w_idx, c0_idx: tvm.select(
                tvm.any(c1_idx * dst_c0 + c0_idx < src_c),
                src(n_idx, h_idx, w_idx, c1_idx * dst_c0 + c0_idx),
                tvm.const(0, src.dtype)),
                name="res_nc1hwc0",
                attrs={"ori_format": "NHWC", "ori_shape": src.shape},
                tag = "NHWC_trans_5HD")
    elif src_format == "NC1HWC0" and dst_format == "NHWC":
        src_n, src_c1, src_hw, src_c0 = tuple(i.value for i in src.shape)
        dst_shape = dst.get("shape")
        dst_tensor = tvm.compute(dst_shape,
            lambda n_idx, hw_idx, c_idx: src(n_idx,
            c_idx // src_c0, hw_idx, c_idx % src_c0),
            name = "res_nhwc",
            tag = "5HD_trans_NHWC")
    elif src_format == "NHWC" and dst_format == "FRACTAL_Z":
        src_n, src_h, src_w, src_c = tuple(i.value for i in src.shape)
        dst_n0 = 16
        dst_n1 = (src_n + dst_n0 - 1) // dst_n0
        dst_c0 = c0_dict[src.dtype]
        dst_c1 = (src_c + dst_c0 - 1) // dst_c0
        dst_shape = dst_c1 * src_h * src_w, dst_n1, dst_n0, dst_c0
        hw = src_h * src_w
        dst_tensor = tvm.compute(
            dst_shape,
            lambda  i, j, k, l: src(j * dst_n0 + k,
            (i % hw) // src_w, (i % hw) // src_w,
            (i // hw) * dst_c0 + l),
            name = "res_fractal_z_weight",
            attrs={"ori_format": "NHWC", "ori_shape": src.shape},
            tag = "NHWC_trans_FZ"
        )
    else:
        raise RuntimeError("not support this kind of format transfer !")

    return dst_tensor