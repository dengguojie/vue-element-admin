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
from impl import trans_data_negative_target_tc
from tbe.dsl.compute import cube_util
from tbe.tvm import api as tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector


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


def _ceil_and_divide(dividend, factor, divisor=16):
    """
    do division and round up to an integer
    """
    if factor == 0 or divisor == 0:
        error_manager_vector.raise_err_specific_reson("trans_data", "Division by zero")
    return (dividend + factor - 1) // factor if factor == divisor else \
        ((dividend + factor - 1) // factor * factor) // divisor


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
    elif src_format.upper() in ("NHWC","ND") and dst_format.upper() in ("NC1HWC0", "FRACTAL_NZ") and \
            (src.get("dtype") == "bfloat16" or tbe_platform.api_check_support("tik.vgatherb")):
        trans_data_positive_source_tc.trans_data_positive_source_tc(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() in ("NC1HWC0", "FRACTAL_NZ") and dst_format.upper() in ("NHWC", "ND") and \
            (src.get("dtype") == "bfloat16" or tbe_platform.api_check_support("tik.vgatherb")):
        trans_data_negative_target_tc.trans_data_negative_target_tc(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "NCHW" and dst_format.upper() == "NC1HWC0" and \
            (src.get("dtype") == "bfloat16" or tbe_platform.api_check_support("tik.vgatherb")):
        trans_data_positive_source_ntc.trans_data_positive_source_ntc(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "NC1HWC0" and dst_format.upper() == "NCHW" and \
            (src.get("dtype") == "bfloat16" or tbe_platform.api_check_support("tik.vgatherb")):
        trans_data_negative_target_ntc.trans_data_negative_target_ntc(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "HWCN" \
            and dst_format.upper() == "FRACTAL_Z" and groups > 1 and groups == src.get("shape")[-1]:
        dst_format = "C1HWNCOC0"
        axis_h, axis_w, axis_c, axis_n = src.get("shape")
        axis_c = axis_c * groups
        axis_n = 1
        src["shape"] = (axis_h, axis_w, axis_c, axis_n)
        depthwise_weight_4d_2_6d(src, dst, src_format, dst_format, kernel_name)
    elif src_format.upper() == "FRACTAL_Z" \
            and dst_format.upper() == "HWCN" and groups > 1 and groups == dst.get("shape")[-1]:
        src_format = "C1HWNCOC0"
        axis_h, axis_w, axis_c, axis_n = dst.get("shape")
        axis_c = axis_c * groups
        axis_n = 1
        axis_c1 = (axis_c + 15) // 16
        src["shape"] = (axis_c1, axis_h, axis_w, axis_n, 16, 16)
        dst["shape"] = (axis_h, axis_w, axis_c, axis_n)
        depthwise_weight_6d_2_4d(src, dst, src_format, dst_format, kernel_name)
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


@tbe_platform.fusion_manager.register("trans_data")
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
    fractal_n0 = 16 # the third params of fractal_nz(d // d0, n // n0, n0, d0)
    def _ceil_div(dividend, divisor):
        if divisor == 0:
            raise RuntimeError("division by zero")
        return (dividend + divisor - 1) // divisor

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
                tag="NHWC_trans_5HD")

    elif src_format == "NC1HWC0" and dst_format == "NHWC":
        src_n, src_c1, src_hw, src_c0 = tuple(i.value for i in src.shape)
        dst_n, dst_h, dst_w, dst_c = dst.get("shape")
        dst_shape = (dst_n, dst_h*dst_w, dst_c)

        if dst_n != src_n:
            error_manager_vector.raise_err_specific_reson("trans_data", "batch should not be changed when trans NC1HWC0 to NHWC!")

        if dst_h*dst_w != src_hw:
            error_manager_vector.raise_err_specific_reson("trans_data", "Ho*Wo should not be changed when trans NC1HWC0 to NHWC!")

        dst_tensor = tvm.compute(
            dst_shape,
            lambda batch_idx, howo_idx, co_idx: src(
                batch_idx, co_idx // src_c0, howo_idx, co_idx % src_c0),
            name="res_nhwc",
            tag="5HD_trans_NHWC")

    elif src_format == "NHWC" and dst_format == "FRACTAL_Z":
        src_n, src_h, src_w, src_c = tuple(i.value for i in src.shape)
        dst_n1 = _ceil_div(src_n, fractal_n0)
        dst_c0 = c0_dict.get(src.dtype)
        dst_c1 = _ceil_div(src_c, dst_c0)
        dst_shape = dst_c1 * src_h * src_w, dst_n1, fractal_n0, dst_c0
        hw = src_h * src_w
        dst_tensor = tvm.compute(
            dst_shape,
            lambda  i, j, k, l: src(j * fractal_n0 + k,
            (i % hw) // src_w, (i % hw) % src_w,
            (i // hw) * dst_c0 + l),
            name="res_fractal_z_weight",
            attrs={"ori_format": "NHWC", "ori_shape": src.shape},
            tag="NHWC_trans_FZ"
        )

    elif src_format == "ND" and dst_format == "FRACTAL_NZ":
        # transform fotmat ND to Nz, the dst_format is Zz actually when input dtype is fp32
        # for the requirement of load2d_transpose
        src_shape = tuple(i.value for i in src.shape)
        block_reduce = c0_dict.get(src.dtype, 16)
        align_factor = block_reduce
        block_size = 16
        if src.dtype == "float32":
            align_factor = block_size
        dst_shape = (
            _ceil_and_divide(src_shape[-1], align_factor, block_reduce),
            _ceil_and_divide(src_shape[-2], align_factor, block_size),
            block_size,
            block_reduce
        )
        d_axis_origin_length = src_shape[-1]
        row_index = -4
        col_index = -3
        if src.dtype == "float32":
            # change Nz shape to Zz shape
            row_index = -3
            col_index = -4
            dst_shape = (dst_shape[-3], dst_shape[-4], dst_shape[-2], dst_shape[-1])
        dst_tensor = tvm.compute(
            dst_shape,
            lambda *indices: tvm.select(
                tvm.all((indices[row_index] * block_reduce + indices[-1]) < d_axis_origin_length),
                src(*indices[:-4],
                    indices[col_index] * block_size + indices[-2],
                    indices[row_index] * block_reduce + indices[-1])
            ),
            name=src.name + "_fractal",
            attrs={"ori_format": "ND", "ori_shape": src.shape, "format": dst_format, "ND_trans_Nz": 1},
            tag="gemm"
        )

    elif src_format == "FRACTAL_NZ" and dst_format == "ND":
        src_shape = tuple(i.value for i in src.shape)
        dst_shape = src.op.attrs["ori_shape"]
        dst_tensor = tvm.compute(
                dst_shape,
                lambda *indices: src(indices[-1] // src_shape[-2],
                indices[-2] // src_shape[-1],
                indices[-2] % src_shape[-1],
                indices[-1] % src_shape[-2]).astype(src.dtype),
                tag="gemm",
                name="res_nd",
                attrs={"ori_format": "FRACTAL_NZ",
                       "ori_shape": src.shape, "Nz_trans_ND": 1})

    elif src_format == "FRACTAL_Z" and dst_format == "NHWC":
        group, src_fkk, src_n, src_c0 = tuple(i.value for i in src.shape)
        dst_shape = dst.get("shape")
        _, hw_length, _ = dst_shape

        dst_tensor = tvm.compute(
            dst_shape,
            lambda n_idx, hw_idx, c_idx:
                # block_dim_reduce, group, fww, n, c0
                src(n_idx // src_n,
                    c_idx // src_c0 * hw_length + hw_idx,
                    n_idx,
                    c_idx % src_c0),
            name="res_nhwc",
            tag="FZ_trans_NHWC"
        )
    else:
        error_manager_vector.raise_err_specific_reson("trans_data", "not support this kind of format transfer !")

    return dst_tensor
