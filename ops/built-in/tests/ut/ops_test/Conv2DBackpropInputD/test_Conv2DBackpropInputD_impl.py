#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
the Conv2DBackpropInputD test
"""
import sys
from math import ceil as math_ceil
from unittest.mock import MagicMock
from unittest.mock import patch

import conv2d_bp_input_ut_testcase
import util_for_conv2d_bp_input as util
from op_test_frame.ut import OpUT
from te import tvm
from te.platform import cce_conf
from te.tvm.target import cce
import tbe
from tbe.dsl import auto_schedule
from te.lang.cce import cce_build_code
from impl.trans_data import trans_data_compute
from impl.conv2d_backprop_input_d import conv2d_backprop_input_d_compute
from impl.conv2d_backprop_input_d import conv2d_backprop_input_d
from tbe.common.context import op_context
from impl.util import platform_adapter

ut_case = OpUT(
    "Conv2DBackpropInputD", "impl.conv2d_backprop_input_d", "conv2d_backprop_input_d"
)

vals = {("CORE_NUM", ): 48,
    ("CUBE_VECTOR_SPLIT",): True,
    ("UB_SIZE", ): 196608,
    ("L0A_SIZE", ): 65536,
    ("L0B_SIZE", ): 65536,
    ("L1_SIZE", ): 524288,
    ("L0C_SIZE", ): 131072,
    ("SOC_VERSION",): "Ascend920A"
}

support_intrinsic_cube_vector_split = {
    ("Intrinsic_fix_pipe_l0c2ub",) : False,
    ("Intrinsic_fix_pipe_l0c2out",) : True,
    ("Intrinsic_data_move_l0c2ub",) : False,
    ("Intrinsic_data_move_l12bt",) : True,
    ("Intrinsic_data_move_ub2l1",) : False,
    ("Intrinsic_mmad", "f162f32",) : True,
    ("CUBE_VECTOR_SPLIT",) : True,
}

DEBUG_MODE = False

def side_effects(*args):
    return vals[args]

def check_intrinsic_cube_vector_split(*args):
    return support_intrinsic_cube_vector_split[args]

def tiling_mock1(*args):
    tiling = {'AL0_matrix': [1, 4, 16, 16, 1, 1], 'AL1_shape': [256, 1, 1, 1],
                    'AUB_channel_wise_flag': None, 'AUB_shape': [256, 2, None, 1],
                    'A_overhead_opt_flag': True, 'BL0_matrix': [4, 6, 16, 16, 1, 1],
                    'BL1_shape': [], 'BUB_channel_wise_flag': None,
                    'BUB_shape': None, 'B_overhead_opt_flag': False, 'CL0_matrix': [6, 1, 16, 16, 1, 1],
                    'CUB_channel_wise_flag': False, 'CUB_matrix': [6, 1, 16, 16, 1, 1],
                    'batch_bef_group_flag': 0, 'block_dim': [3, 1, 1, 1],
                    'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1,
                    'BL0_pbuffer': 2, 'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2,
                    'CUB_pbuffer': 2, 'UBG_pbuffer': 2}, 'n_bef_batch_flag': 0, 'n_bef_group_flag': 0,
                    'tbe_compile_para': 0}
    return tiling


def tiling_mock2(*args):
    tiling = {'AL0_matrix': [1, 4, 16, 16, 1, 1], 'AL1_shape': [256, 1, 1, 1],
                    'AUB_channel_wise_flag': None, 'AUB_shape': [256, 2, None, 1],
                    'A_overhead_opt_flag': False, 'BL0_matrix': [4, 6, 16, 16, 1, 1],
                    'BL1_shape': [], 'BUB_channel_wise_flag': None,
                    'BUB_shape': None, 'B_overhead_opt_flag': True, 'CL0_matrix': [6, 1, 16, 16, 1, 1],
                    'CUB_channel_wise_flag': False, 'CUB_matrix': [6, 1, 16, 16, 1, 1],
                    'batch_bef_group_flag': 0, 'block_dim': [3, 1, 1, 1],
                    'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1,
                    'BL0_pbuffer': 2, 'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2,
                    'CUB_pbuffer': 2, 'UBG_pbuffer': 2}, 'n_bef_batch_flag': 0, 'n_bef_group_flag': 0,
                    'tbe_compile_para': 0}
    return tiling

def tiling_mock_preload_cl0c_al1(*args):
    tiling = {'AL0_matrix': [13, 3, 16, 16, 1, 1], 'AL1_shape': [2304, 1, 1, 1],
                    'AUB_channel_wise_flag': None, 'AUB_shape': None,
                    'A_overhead_opt_flag': 0, 'BL0_matrix': [3, 8, 16, 16, 1, 1],
                    'BL1_shape': [288, 1, 1, 1], 'BUB_channel_wise_flag': None,
                    'BUB_shape': None, 'B_overhead_opt_flag': 0, 'CL0_matrix': [8, 13, 16, 16, 1, 1],
                    'CUB_channel_wise_flag': False, 'CUB_matrix': [2, 13, 16, 16, 1, 1],
                    'batch_bef_group_flag': 0, 'block_dim': [16, 2, 1, 1],
                    'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1,
                    'BL0_pbuffer': 2, 'BL1_pbuffer': 2, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2,
                    'CUB_pbuffer': 2, 'UBG_pbuffer': 2}, 'n_bef_batch_flag': 0, 'n_bef_group_flag': 0,
                    'tbe_compile_para': 24}
    return tiling

def _gen_kernel_name(dedy_shape, w_shape, dx_shape, strides, data_flow):
    dedy_shape_info = "_".join([str(i) for i in dedy_shape])
    w_shape_info = "_".join([str(i) for i in w_shape])
    dx_shape_info = "_".join([str(i) for i in dx_shape])
    stride_shape_info = "_".join([str(i) for i in strides])

    kernel_name = "Conv2DBackpropInputD_dy_{}_w_{}_x_{}_stride_{}_data_flow_{}".format(
        dedy_shape_info, w_shape_info, dx_shape_info, stride_shape_info, data_flow
    )
    return kernel_name


def _gen_trans_data_case(
    w_dtype,
    dedy_dtype,
    dx_dtype,
    w_shape,
    dedy_shape,
    dx_shape,
    w_format,
    dedy_format,
    dx_format,
    input_size,
    stride,
    padding,
    dilations=(1, 1, 1, 1),
    groups=1,
    expect="success",
    data_flow="default",
):

    kernel_name = _gen_kernel_name(dedy_shape, w_shape, dx_shape, stride, data_flow)

    dedy = {
        "shape": util.shape_4d_to_5hd(dedy_shape, dedy_dtype, dedy_format),
        "dtype": dedy_dtype,
        "format": dedy_format,
        "ori_shape": dedy_shape,
        "ori_format": dedy_format,
    }

    filters = {
        "shape": util.shape_4d_to_fz(w_shape, w_dtype, w_format),
        "dtype": w_dtype,
        "format": w_format,
        "ori_shape": util.get_ori_shape(w_shape, w_dtype, w_format),
        "ori_format": w_format,
    }

    dedx = {
        "shape": util.shape_4d_to_5hd(dx_shape, dx_dtype, dx_format),
        "dtype": dx_dtype,
        "format": dx_format,
        "ori_shape": dx_shape,
        "ori_format": dx_format,
    }

    input_sizes = input_size
    strides = stride
    padding = util.gen_padding_size(dx_shape, w_shape, padding, stride, dilations)
    data_format = dedy_format

    if DEBUG_MODE:
        print(
            kernel_name,
            [
                filters,
                dedy,
                dedx,
                input_sizes,
                strides,
                padding,
                dilations,
                groups,
                data_format,
            ],
        )

    return {
        "params": [
            filters,
            dedy,
            dedx,
            input_sizes,
            strides,
            padding,
            dilations,
            groups,
            data_format,
        ],
        "case_name": kernel_name,
        "expect": expect,
        "format_expect": [],
        "support_expect": True,
    }


def _gen_conv2d_bp_input_op_case():
    for test_case in conv2d_bp_input_ut_testcase.conv2d_bp_input_op_testcase:
        ut_case.add_case(["Ascend910A"], _gen_trans_data_case(*test_case))

def _test_op_check_supported(test_arg):
    from impl.conv2d_backprop_input_d import check_supported
    weight = {"ori_shape": (32, 32, 3, 3), "dtype": "float16", "ori_format": "NCHW"}
    out_backprop = {"ori_shape": (16, 32, 2, 2), "dtype": "float16", "ori_format": "NCHW"}
    y = {"ori_shape": (16, 32, 5, 5), "dtype": "float16", "ori_format": "NCHW"}
    input_size = (16, 32, 5, 5)
    check_supported(weight, out_backprop, y, input_size, (1, 1, 2, 2), (0, 0, 0, 0),
                    dilations=(1, 1, 1, 1), groups=1, data_format="NCHW",
                    kernel_name="conv2d_backprop_input")

def _gen_conv2d_bp_input_check_support_case():
    ut_case.add_cust_test_func("Ascend910A", test_func=_test_op_check_supported)

def _test_nhwc_in_nhwc_out_case_1(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
            conv_filter = (160, 1, 1, 160)
            out_backprop = (1, 64, 128, 160)
            input_size = (1, 64, 128, 160)
            strides = (1, 1)
            pads = (0, 0, 0, 0)
            dilations = (1, 1, 1, 1)
            data_type = "float16"
            with cce():
                fmap_ori = tvm.placeholder(out_backprop, name="fmap_ori", dtype=data_type)
                weight_ori = tvm.placeholder(conv_filter, name="weight_ori", dtype=data_type)
                fmap = trans_data_compute(fmap_ori, None, src_format="NHWC", dst_format="NC1HWC0")
                weight = trans_data_compute(weight_ori, None, src_format = "NHWC", dst_format="FRACTAL_Z")
                y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NHWC", "format" : "NC1HWC0"}
                conv_res = conv2d_backprop_input_d_compute(weight, fmap, y, input_size, strides, pads)
                src_n, src_c1, src_hw, src_c0 =  tuple(i.value for i in conv_res.shape)
                out = trans_data_compute(conv_res, {"shape" : (src_n, src_hw, src_c1*src_c0)}, src_format="NC1HWC0",
                                        dst_format="NHWC")
                tensor_list = [weight_ori, fmap_ori, out]
                sch = auto_schedule(out)
                config = {
                    "print_ir" : False,
                    "need_build" : True,
                    "name" : "conv2d_bp_input_ut_testcase_1",
                    "tensor_list" : tensor_list
                }
                cce_build_code(sch, config)

def _test_nhwc_in_nhwc_out_case_2(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
            conv_filter = (256, 3, 3, 256)
            out_backprop = (2, 14, 14, 256)
            input_size = (2, 28, 28, 256)
            strides = (2, 2)
            pads = (0, 1, 0, 1)
            dilations = (1, 1, 1, 1)
            out_backprop_after_dialtion = (2, 27, 27, 256)
            strides_after_dialtion = (1, 1)
            pads_after_dilation = (0, 1, 0, 1)
            data_type = "float16"
            with cce():
                fmap_ori = tvm.placeholder(out_backprop_after_dialtion, name="fmap_ori", dtype=data_type)
                weight_ori = tvm.placeholder(conv_filter, name="weight_ori", dtype=data_type)
                fmap = trans_data_compute(fmap_ori, None, src_format="NHWC", dst_format="NC1HWC0")
                weight = trans_data_compute(weight_ori, None, src_format = "NHWC", dst_format="FRACTAL_Z")
                y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NHWC", "format" : "NC1HWC0"}
                conv_res = conv2d_backprop_input_d_compute(weight, fmap, y, input_size, strides_after_dialtion, pads_after_dilation)
                src_n, src_c1, src_hw, src_c0 =  tuple(i.value for i in conv_res.shape)
                out = trans_data_compute(conv_res, {"shape" : (src_n, src_hw, src_c1*src_c0)}, src_format="NC1HWC0",
                                        dst_format="NHWC")
                tensor_list = [weight_ori, fmap_ori, out]
                sch = auto_schedule(out)
                config = {
                    "print_ir" : False,
                    "need_build" : True,
                    "name" : "conv2d_bp_input_ut_testcase_2",
                    "tensor_list" : tensor_list
                }
                cce_build_code(sch, config)

def _test_set2d_case_1(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
            filter_frac = (144, 16, 16, 16)
            out_shape_5hd = (2, 16, 14, 14, 16)
            input_size = (2, 256, 28, 28)
            strides = (2, 2)
            pads = (0, 1, 0, 1)
            dilations = (1, 1, 1, 1)
            data_type = "float16"
            with cce():
                weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                        attrs={"ori_shape": (256, 256, 3, 3), "dtype":data_type, "ori_format": "NCHW"})
                dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                    attrs={"ori_shape": (2, 256, 14, 14), "dtype":data_type, "ori_format": "NCHW"})
                y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW"}
                out = conv2d_backprop_input_d_compute(weight, dedy, y, input_size, strides, pads)
                tensor_list = [weight, dedy, out]
                sch = auto_schedule(out)
                config = {
                    "name" : "conv2d_bp_input_ut_set2d_case_1",
                    "tensor_list" : tensor_list
                }
                cce_build_code(sch, config)

def _gen_conv2d_bp_input_920A_case():
    ut_case.add_cust_test_func(test_func=_test_nhwc_in_nhwc_out_case_1)
    ut_case.add_cust_test_func(test_func=_test_nhwc_in_nhwc_out_case_2)
    ut_case.add_cust_test_func(test_func=_test_set2d_case_1)

def _gen_conv2d_bp_input_cube_vector_split_case_mock(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("impl.util.platform_adapter.tbe_platform.get_soc_spec", MagicMock(side_effect=side_effects)):
            with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
                with cce():
                    x = tvm.placeholder((2, 7, 7, 32), name="x", dtype="float16", attrs={"ori_shape": (2, 7, 7, 32), "format": "NHWC", "ori_format": "NHWC"})
                    x_trans = trans_data_compute(x, None, "NHWC", "NC1HWC0")
                    weight = tvm.placeholder((18, 2, 16, 16), name="filter", dtype="float16", attrs={"ori_shape": (3, 3, 32, 32), "format": "FRACTAL_Z", "ori_format": "HWCN"})
                    y = {"shape": (2, 2, 14, 14, 16), "ori_shape": (2, 14, 14, 32), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
                    dx_res = conv2d_backprop_input_d_compute(weight, x_trans, y, (2, 14, 14, 32), (1, 2, 2, 1), (0, 1, 0, 1))
                    trans_out = {"shape": (2, 14, 14, 32), "ori_shape": (2, 14, 14, 32), "format": "NHWC", "ori_format": "NHWC", "dtype": "float16"}
                    out = trans_data_compute(dx_res, trans_out, "NC1HWC0", "NHWC")
                    sch = auto_schedule(out)

def _test_conv2d_bp_input_bf16_case_1(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("impl.util.platform_adapter.tbe_platform.get_soc_spec", MagicMock(side_effect=side_effects)):
            with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
                with cce():
                    x = tvm.placeholder((2, 7, 7, 32), name="x", dtype="bfloat16",
                                        attrs={
                                            "ori_shape": (2, 7, 7, 32), "format": "NHWC", 
                                            "ori_format": "NHWC"
                                        })
                    x_5hd = trans_data_compute(x, None, "NHWC", "NC1HWC0")
                    weight = tvm.placeholder((18, 2, 16, 16), name="filter", dtype="bfloat16",
                                                attrs={
                                                    "ori_shape": (3, 3, 32, 32),
                                                    "format": "FRACTAL_Z",
                                                    "ori_format": "HWCN"
                                                })
                    y = {"shape": (2, 2, 14, 14, 16), "ori_shape": (2, 14, 14, 32), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "bfloat16"}
                    dx = conv2d_backprop_input_d_compute(weight, x_5hd, y, (2, 14, 14, 32), (1, 2, 2, 1), (0, 1, 0, 1))
                    trans_out = {"shape": (2, 14, 14, 32), "ori_shape": (2, 14, 14, 32), "format": "NHWC", "ori_format": "NHWC", "dtype": "bfloat16"}
                    out = trans_data_compute(dx,trans_out, "NC1HWC0", "NHWC")
                    sch = auto_schedule(out)

def _test_conv2d_bp_input_fp32_case_1(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("impl.util.platform_adapter.tbe_platform.get_soc_spec", MagicMock(side_effect=side_effects)):
            with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
                with cce():
                    x = tvm.placeholder((16, 2, 2, 32), name="x", dtype="float32",
                                        attrs={
                                            "ori_shape": (16, 2, 2, 32), "format": "NHWC", 
                                            "ori_format": "NHWC"
                                        })
                    x_5hd = trans_data_compute(x, None, "NHWC", "NC1HWC0")
                    weight = tvm.placeholder((36, 2, 16, 8), name="filter", dtype="float32",
                                                attrs={
                                                    "ori_shape": (3, 3, 32, 32),
                                                    "format": "FRACTAL_Z",
                                                    "ori_format": "HWCN"
                                                })
                    y = {"shape": (16, 4, 4, 4, 8), "ori_shape": (16, 4, 4, 32), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float32"}
                    dx = conv2d_backprop_input_d_compute(weight, x_5hd, y, (16, 4, 4, 32), (1, 1), "VALID", (1, 1, 1, 1))
                    trans_out = {"shape": (16, 4, 4, 32), "ori_shape": (16, 4, 4, 32), "format": "NHWC", "ori_format": "NHWC", "dtype": "float32"}
                    out = trans_data_compute(dx, trans_out, "NC1HWC0", "NHWC")
                    sch = auto_schedule(out)

def _test_conv2d_bp_input_fp32_case_2(test_arg):
   with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("impl.util.platform_adapter.tbe_platform.get_soc_spec", MagicMock(side_effect=side_effects)):
            with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
                with cce():
                    x = tvm.placeholder((2, 7, 7, 2048), name="x", dtype="float32",
                                        attrs={
                                            "ori_shape": (2, 7, 7, 2048), "format": "NHWC", 
                                            "ori_format": "NHWC"
                                        })
                    x_5hd = trans_data_compute(x, None, "NHWC", "NC1HWC0")
                    weight = tvm.placeholder((64, 128, 16, 8), name="filter", dtype="float32",
                                                attrs={
                                                    "ori_shape": (1, 1, 512, 2048),
                                                    "format": "FRACTAL_Z",
                                                    "ori_format": "HWCN"
                                                })
                    y = {"shape": (2, 64, 7, 7, 8), "ori_shape": (2, 7, 7, 512), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float32"}
                    dx = conv2d_backprop_input_d_compute(weight, x_5hd, y, (2, 7, 7, 512), (1, 1), "VALID", (1, 1, 1, 1))
                    trans_out = {"shape": (2, 7, 7, 512), "ori_shape": (2, 7, 7, 512), "format": "NHWC", "ori_format": "NHWC", "dtype": "float32"}
                    out = trans_data_compute(dx, trans_out, "NC1HWC0", "NHWC")
                    sch = auto_schedule(out)

def _test_conv2d_bp_input_hf32_case_1(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("impl.util.platform_adapter.tbe_platform.get_soc_spec", MagicMock(side_effect=side_effects)):
            with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
                with cce():
                    x = tvm.placeholder((16, 2, 2, 32), name="x", dtype="float32",
                                        attrs={
                                            "ori_shape": (16, 2, 2, 32), "format": "NHWC", 
                                            "ori_format": "NHWC"
                                        })
                    x_5hd = trans_data_compute(x, None, "NHWC", "NC1HWC0")
                    weight = tvm.placeholder((36, 2, 16, 8), name="filter", dtype="float32",
                                                attrs={
                                                    "ori_shape": (3, 3, 32, 32),
                                                    "format": "FRACTAL_Z",
                                                    "ori_format": "HWCN"
                                                })
                    y = {"shape": (16, 4, 4, 4, 8), "ori_shape": (16, 4, 4, 32), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float32"}
                    dx = conv2d_backprop_input_d_compute(weight, x_5hd, y, (16, 4, 4, 32), (1, 1), "VALID", (1, 1, 1, 1), impl_mode="high_performance")
                    trans_out = {"shape": (16, 4, 4, 32), "ori_shape": (16, 4, 4, 32), "format": "NHWC", "ori_format": "NHWC", "dtype": "float32"}
                    out = trans_data_compute(dx, trans_out, "NC1HWC0", "NHWC")
                    sch = auto_schedule(out)

def _test_conv2d_bp_input_hf32_case_2(test_arg):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("impl.util.platform_adapter.tbe_platform.get_soc_spec", MagicMock(side_effect=side_effects)):
            with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
                with cce():
                    x = tvm.placeholder((2, 7, 7, 2048), name="x", dtype="float32",
                                        attrs={
                                            "ori_shape": (2, 7, 7, 2048), "format": "NHWC", 
                                            "ori_format": "NHWC"
                                        })
                    x_5hd = trans_data_compute(x, None, "NHWC", "NC1HWC0")
                    weight = tvm.placeholder((64, 128, 16, 8), name="filter", dtype="float32",
                                                attrs={
                                                    "ori_shape": (1, 1, 512, 2048),
                                                    "format": "FRACTAL_Z",
                                                    "ori_format": "HWCN"
                                                })
                    y = {"shape": (2, 64, 7, 7, 8), "ori_shape": (2, 7, 7, 512), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float32"}
                    dx = conv2d_backprop_input_d_compute(weight, x_5hd, y, (2, 7, 7, 512), (1, 1), "VALID", (1, 1, 1, 1), impl_mode="high_performance")
                    trans_out = {"shape": (2, 7, 7, 512), "ori_shape": (2, 7, 7, 512), "format": "NHWC", "ori_format": "NHWC", "dtype": "float32"}
                    out = trans_data_compute(dx, trans_out, "NC1HWC0", "NHWC")
                    sch = auto_schedule(out)

def _test_conv2d_bp_input_allocate_at1(test_arg):
    with patch("tbe.common.tiling.tiling_api.get_tiling", MagicMock(side_effect=tiling_mock1)):
        with cce():
            x_5hd = tvm.placeholder((1, 1, 1, 2, 16), name="x", dtype="float16",
                                    attrs={"ori_shape": (1, 1, 2, 16),
                                           "format": "NC1HWC0",
                                           "ori_format": "NHWC"})
            weight = tvm.placeholder((16*6, 1, 16, 16), name="filter", dtype="float16",
                                     attrs={"ori_shape": (4, 4, 96, 16),
                                            "format": "FRACTAL_Z",
                                            "ori_format": "HWCN"})
            y = {"shape": (1, 6, 2, 4, 16), "ori_shape": (1, 2, 4, 96), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
            dx = conv2d_backprop_input_d_compute(weight, x_5hd, y, (1, 2, 4, 96), (2, 2), "SAME", (1, 1, 1, 1))
            with op_context.OpContext():
                sch = auto_schedule(dx)
            config = {"name": "dx", "tensor_list":[weight, x_5hd, dx]}
            platform_adapter.tbe.build(sch, config)

def _test_conv2d_bp_input_allocate_at2(test_arg):
    with patch("tbe.common.tiling.tiling_api.get_tiling", MagicMock(side_effect=tiling_mock2)):
        with cce():
            x_5hd = tvm.placeholder((1, 1, 1, 2, 16), name="x", dtype="float16",
                                    attrs={"ori_shape": (1, 1, 2, 16),
                                           "format": "NC1HWC0",
                                           "ori_format": "NHWC"})
            weight = tvm.placeholder((16*6, 1, 16, 16), name="filter", dtype="float16",
                                     attrs={"ori_shape": (4, 4, 96, 16),
                                            "format": "FRACTAL_Z",
                                            "ori_format": "HWCN"})
            y = {"shape": (1, 6, 2, 4, 16), "ori_shape": (1, 2, 4, 96), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
            dx = conv2d_backprop_input_d_compute(weight, x_5hd, y, (1, 2, 4, 96), (2, 2), "SAME", (1, 1, 1, 1))
            with op_context.OpContext():
                sch = auto_schedule(dx)
            config = {"name": "dx", "tensor_list":[weight, x_5hd, dx]}
            platform_adapter.tbe.build(sch, config)

def _test_conv2d_bp_input_preload_cl0c_al1(test_arg):
    with patch("tbe.common.tiling.tiling_api.get_tiling", MagicMock(side_effect=tiling_mock_preload_cl0c_al1)):
        weight = {'shape': (144, 16, 16, 16), 'ori_shape': (3, 3, 256, 256), 'format': 'FRACTAL_Z', 'ori_format': 'HWCN', 'dtype': 'float16'}
        out_backprop = {'shape': (256, 16, 14, 14, 16), 'ori_shape': (256, 14, 14, 256), 'format': 'NC1HWC0', 'ori_format': 'NHWC', 'dtype': 'float16'}
        y = {'shape': (256, 16, 14, 14, 16), 'ori_shape': (256, 14, 14, 256), 'format': 'NC1HWC0', 'ori_format': 'NHWC', 'dtype': 'float16'}
        input_size = (256, 14, 14, 256)
        strides = (1, 1, 1, 1)
        pads = (1, 1, 1, 1)
        dilations = (1, 1, 1, 1)
        groups = 1
        data_format = 'NHWC'
        conv2d_backprop_input_d(weight, out_backprop, y, input_size, strides, pads, dilations, groups, data_format)


_gen_conv2d_bp_input_op_case()
_gen_conv2d_bp_input_check_support_case()
ut_case.add_cust_test_func(test_func=_gen_conv2d_bp_input_cube_vector_split_case_mock)
#_gen_conv2d_bp_input_920A_case()
ut_case.add_cust_test_func(test_func=_test_conv2d_bp_input_bf16_case_1)
ut_case.add_cust_test_func(test_func=_test_conv2d_bp_input_fp32_case_1)
ut_case.add_cust_test_func(test_func=_test_conv2d_bp_input_fp32_case_2)
ut_case.add_cust_test_func(test_func=_test_conv2d_bp_input_hf32_case_1)
ut_case.add_cust_test_func(test_func=_test_conv2d_bp_input_hf32_case_2)
ut_case.add_cust_test_func(test_func=_test_conv2d_bp_input_allocate_at1)
ut_case.add_cust_test_func(test_func=_test_conv2d_bp_input_allocate_at2)
ut_case.add_cust_test_func(test_func=_test_conv2d_bp_input_preload_cl0c_al1)



if __name__ == "__main__":
    ut_case.run("Ascend910A")
    sys.exit(0)
