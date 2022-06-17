"""
Copyright (C) 2019-2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

trans_data
"""
from __future__ import absolute_import
import copy
from tbe.dsl.base.operation import get_te_var
from impl.util import fusion_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.dynamic.transpose import Transpose
from . import trans_data_positive_source_tc
from . import trans_data_negative_target_ntc
from . import trans_data_positive_source_ntc
from . import trans_data_negative_target_tc
from . import conv2d_data_rm_compute

# the NCHW format length
NCHW_LENTH = 4


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    TILING_MAX_SIZE_GM = 2048  # 16KB
    MAX_INT64_VALUE = 2 ** 64 - 1


def is_do_with_transpose_formats(src_format, dst_format):
    """
    judge src_format and dst_format in the list: ["NCHW", "NHWC", "HWCN", "CHWN"]
    """
    format_list = ["NCHW", "NHWC", "HWCN", "CHWN"]
    if src_format in format_list and dst_format in format_list and src_format != dst_format:
        return True
    return False


def is_do_with_positive_source_ntc_100(src_format, dst_format):
    """
    judge src_format and dst_format in the dict:
    {"NCDHW":"NDC1HWC0", "NCHW":"NC1HWC0", "HWCN":"FRACTAL_Z", "HWCN":"FRACTAL_ZN", "DHWCN":"FRACTAL_Z_3D",
    "ND":"FRACTAL_Z", "ND":"FRACTAL_ZN", "NCHW":"FRACTAL_Z", "NCHW":"FRACTAL_ZN", "NCDHW":"FRACTAL_Z_3D"}
    """
    support_src_dst_formats = {"NCDHW": ["NDC1HWC0", "FRACTAL_Z_3D"], "HWCN": ["FRACTAL_Z", "FRACTAL_ZN"],
                               "DHWCN": ["FRACTAL_Z_3D"], "ND": ["FRACTAL_Z", "FRACTAL_ZN"],
                               "NCHW": ["FRACTAL_Z", "FRACTAL_ZN", "NC1HWC0"]}
    if src_format in support_src_dst_formats and dst_format in support_src_dst_formats.get(src_format):
        return True
    return False


# 'pylint: disable=unused-argument, too-many-arguments, too-many-locals, too-many-boolean-expressions
# 'pylint: disable=inconsistent-return-statements
@register_operator("TransData")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def trans_data(src, dst, src_format=None, dst_format=None, group=1, kernel_name="trans_data"):
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
    group: int
        default 1
    kernel_name: str
        kernel name, default value is "format_transfer"

    Returns
    -------
    None
    """
    if src_format is None:
        src_format = src.get("format").upper().split(":")[0]
    else:
        src_format = src_format.upper()

    if dst_format is None:
        dst_format = dst.get("format").upper().split(":")[0]
    else:
        dst_format = dst_format.upper()
    tbe_context.get_context().add_build_res("pattern", "TransData")
    if ((src_format == "NC1HWC0" and dst_format == "NHWC") or
        (src_format == "FRACTAL_NZ" and dst_format in ("ND", "NHWC", "NCHW", "NC1HWC0")) or
        (src_format == "FRACTAL_Z_3D" and dst_format == "NDHWC") or
        (src_format == "NDC1HWC0" and dst_format == "NDHWC")):
        trans_data_negative_target_tc.trans_data_negative_target_tc(src, dst, src_format, dst_format, kernel_name)
    elif (((src_format == "NC1HWC0" and dst_format == "NCHW") or
           (src_format == "FRACTAL_Z_3D" and dst_format == "NCDHW") or
           (src_format == "NDC1HWC0" and dst_format == "NCDHW") or
           ((src_format in ("FRACTAL_Z", "FRACTAL_ZN")) and (dst_format == "HWCN")) or
           ((src_format in ("FRACTAL_Z", "FRACTAL_ZN")) and (dst_format == "NCHW")) or
           ((src_format in ("FRACTAL_Z", "FRACTAL_ZN")) and (dst_format == "ND")) or
           (src_format == "FRACTAL_Z_3D" and dst_format == "DHWCN"))):
        trans_data_negative_target_ntc.trans_data_negative_target_ntc(src, dst, src_format, dst_format, kernel_name)
    elif is_do_with_transpose_formats(src_format, dst_format):
        x_dtype = src.get("dtype").lower()
        y_dtype = dst.get("dtype").lower()
        tik_inst = tik.Tik()
        data_in = tik_inst.Tensor(x_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_in")
        data_out = tik_inst.Tensor(y_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_out")
        data_workspace = tik_inst.Tensor(y_dtype, (1024, ), tik.scope_gm, "data_workspace", is_workspace=True)
        data_tiling = tik_inst.Tensor("int64", (Constant.TILING_MAX_SIZE_GM,), tik.scope_gm, "data_tiling")
        tensor_list = [data_in, None, data_out, data_workspace, data_tiling]
        input_list = [data_in]
        transpose_instance = Transpose(tik_inst, x_dtype, tensor_list, kernel_name)
        return transpose_instance.compute(input_list)
    elif is_do_with_positive_source_ntc_100(src_format, dst_format):
        trans_data_positive_source_ntc.trans_data_positive_source_ntc(src, dst, src_format, dst_format, kernel_name)
    else:
        trans_data_positive_source_tc.trans_data_positive_source_tc(src, dst, src_format, dst_format, kernel_name)


def _nchw_to_5hd(input_x):
    """
    trans nchw to nc1hwc0
    """
    input_dtype = input_x.dtype
    input_shape = shape_util.shape_to_list(input_x.shape)
    if len(input_shape) == NCHW_LENTH:
        shape_n, shape_c, shape_h, shape_w = input_shape
    else:
        shape_n, shape_c, shape_h, shape_w = shape_util.shape_to_list(input_x.op.attrs["shape"])
    shape_c0 = tbe_platform.CUBE_MKN[input_dtype]["mac"][1]
    shape_c1 = (shape_c + shape_c0 - 1) // shape_c0

    input_align_shape = (shape_n, shape_c1 * shape_c0, shape_h * shape_w)
    reshape_shape = (shape_n, shape_c1, shape_c0, shape_h * shape_w)
    transpose_shape = (shape_n, shape_c1, shape_h * shape_w, shape_c0)
    output_shape = (shape_n, shape_c1, shape_h, shape_w, shape_c0)
    output_attrs = copy.deepcopy(input_x.op.attrs)
    output_attrs["shape"] = output_shape

    if len(input_shape) == NCHW_LENTH:
        input_ub = tvm.compute(input_align_shape,
                               lambda n, c, hw: tvm.select(c < shape_c, input_x(n, c, hw // shape_w, hw % shape_w)),
                               name="input_ub_td"
                              )
    else:
        input_ub = tvm.compute(input_align_shape,
                               lambda n, c, hw: tvm.select(c < shape_c, input_x(n, c, hw)),
                               name="input_ub_td"
                              )
    input_ub_pad = tvm.compute(input_align_shape,
                               lambda n, c, hw: tvm.select(c >= shape_c, tvm.const(0, input_dtype)),
                               name="input_ub_pad"
                              )
    input_ub_vn = tvm.compute(input_align_shape,
                              lambda n, c, hw: input_ub(n, c, hw) + input_ub_pad(n, c, hw),
                              name="input_ub_vn"
                             )
    reshape_c = tvm.compute(reshape_shape,
                            lambda n, c1, c0, hw: input_ub_vn(n, c1*shape_c0 + c0, hw),
                            name="reshape_c"
                           )
    transpose_hw_c0 = tvm.compute(transpose_shape,
                                  lambda n, c1, hw, c0: reshape_c(n, c1, c0, hw),
                                  name="transpose_hw_c0"
                                 )
    res = tvm.compute(output_shape,
                      lambda n, c1, h, w, c0: transpose_hw_c0(n, c1, h*shape_w + w, c0),
                      name="split_hw",
                      tag="NCHW_trans_5HD",
                      attrs=output_attrs
                      )
    return res


def _nc1hwc0_to_nchw(src, dst):
    """
    algorithm: trans nc1hwc0 to nchw

    Parameters
    ----------
    src : Tensor, Tensor of input

    dst: dict, shape and dtype of output, should be same shape and type as input

    Returns
    -------
    Tensor
    """
    src_n, src_c1, src_hw, src_c0 = src.shape
    remove_pad_flag = False
    if src.op.tag == "conv2d_backprop_input":
        real_c = get_te_var("dx_c").get_tvm_var()
    elif src.op.name == "invalid_conv2d_rmpad":
        real_c = get_te_var("c_out").get_tvm_var()
    elif src.op.tag == "convolution_C":
        real_c = get_te_var("c_out").get_tvm_var()
        src = src.op.input_tensors[0]
        remove_pad_flag = True
    else:
        real_c = dst.get("ori_shape")[1]
    transpose_shape = (src_n, src_c1, src_c0, src_hw)
    transpose_tensor = tvm.compute(
        transpose_shape,
        lambda n_idx, c1_idx, c0_idx, hw_idx:
            src(n_idx, c1_idx, hw_idx, c0_idx),
        name="transpose")
    dst_shape = (src_n, real_c, src_hw)
    dst_tensor = tvm.compute(
        dst_shape,
        lambda n_idx, c_idx, hw_idx:
            transpose_tensor(n_idx, c_idx // src_c0, c_idx % src_c0, hw_idx),
        name="res_nchw",
        tag="5HD_TRANS_NCHW")

    if remove_pad_flag:
        dst_tensor = conv2d_data_rm_compute(dst_tensor)

    return dst_tensor


@register_operator_compute("TransData", op_mode="dynamic", support_fusion=True)
def trans_data_fusion_compute(src, dst, src_format=None, dst_format=None, group=1, kernel_name="trans_data"):
    """
    algorithm: format_transfer
    used for format transformation , only support transfer between NHWC and NC1HWC0
    Parameters
    ----------
    src : tvm.tensor
    input_tenor
    dst: dict
    shape and dtype of output, should be same shape and type as input
    src_format: str
    source data format, can be NCHW and NC1HWC0, default value is None
    dst_format: str
    target data format, can be NC1HWC0 and NCHW, default value is None
    group: int
    default 1
    kernel_name: str
    kernel name, default value is "trans_data"

    Returns
    -------
    then tensor after transformation
    """
    if src_format is None:
        src_format = src.op.attrs["format"].value.upper().split(":")[0]
    else:
        src_format = src_format.upper()

    if dst_format is None:
        dst_format = dst.get("format").upper().split(":")[0]
    else:
        dst_format = dst_format.upper()

    fusion_util.check_fusion_input([src])
    if src.dtype != "float16":
        error_manager_vector.raise_err_specific_reson(
            "trans_data", "only support float16"
        )

    if src_format == "NCHW" and dst_format == "NC1HWC0":
        return _nchw_to_5hd(src)
    elif src_format == "NC1HWC0" and dst_format == "NCHW":
        return _nc1hwc0_to_nchw(src, dst)
    else:
        error_manager_vector.raise_err_specific_reson(
            "trans_data", "only support format transfer between NCHW and NC1HWC0"
        )
