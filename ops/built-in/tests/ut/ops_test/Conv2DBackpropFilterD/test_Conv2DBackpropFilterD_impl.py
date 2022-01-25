#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import conv2d_bp_filter_ut_testcase
import util_for_conv2d_bp_filter as util
from op_test_frame.ut import OpUT
from impl.conv2d_backprop_filter_d import get_op_support_info
from impl.conv2d_backprop_filter_d import conv2d_backprop_filter_compute
from te import tvm
from impl.util.platform_adapter import tbe
from te.platform.cce_conf import te_set_version
from impl.trans_data import trans_data_compute
from tbe.common.context import op_context
from test_mock_case import *


ut_case = OpUT(
    "conv2d_backprop_filter_d",
    "impl.conv2d_backprop_filter_d",
    "conv2d_backprop_filter_d",
)
DEBUG_MODE = False


vals = {("CORE_NUM", ): 48,
        ("CUBE_VECTOR_SPLIT",): True,
        ("UB_SIZE", ): 196608,
        ("L0A_SIZE", ): 65536,
        ("L0B_SIZE", ): 65536,
        ("L1_SIZE", ): 524288,
        ("L0C_SIZE", ): 131072,
        ("Intrinsic_fix_pipe_l0c2out",): True,
        ("Compiler_arch",): "dav-c220-cube",
        ("AICORE_TYPE",): "AiCore",
        ("SOC_VERSION",): "Ascend920A",
        ("Intrinsic_fix_pipe_unit_list",): True,
        ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): True
        }

def get_soc_mock(*args):
    return vals[args]


def tiling_mock(*args):
    tiling = {'AL0_matrix': [1, 2, 16, 8, 1, 1], 'AL1_shape': [64, 1, 1, 1],
                    'AUB_channel_wise_flag': None, 'AUB_shape': [1, 0, 0, 0],
                    'A_overhead_opt_flag': 0, 'BL0_matrix': [2, 1, 16, 16, 1, 1],
                    'BL1_shape': [64, 1, 1, 1], 'BUB_channel_wise_flag': None,
                    'BUB_shape': None, 'B_overhead_opt_flag': 0, 'CL0_matrix': [1, 1, 16, 16, 1, 1],
                    'CUB_channel_wise_flag': False, 'CUB_matrix': [1, 1, 16, 16, 1, 1],
                    'batch_bef_group_flag': 0, 'block_dim': [1, 1, 1, 1],
                    'manual_pingpong_buffer': {'AL0_pbuffer': 1, 'AL1_pbuffer': 1, 'AUB_pbuffer': 1,
                    'BL0_pbuffer': 1, 'BL1_pbuffer': 1, 'BUB_pbuffer': 1, 'CL0_pbuffer': 1,
                    'CUB_pbuffer': 1, 'UBG_pbuffer': 1}, 'n_bef_batch_flag': 0, 'n_bef_group_flag': 0,
                    'tbe_compile_para': 0}
    return tiling


def tiling_mock2(*args):
    tiling = {'AL0_matrix': [1, 1, 16, 8, 1, 1], 'AL1_shape': [64, 1, 1, 1],
                    'AUB_channel_wise_flag': None, 'AUB_shape': [1, 0, 0, 0],
                    'A_overhead_opt_flag': 0, 'BL0_matrix': [1, 1, 16, 16, 1, 1],
                    'BL1_shape': [64, 1, 1, 1], 'BUB_channel_wise_flag': None,
                    'BUB_shape': None, 'B_overhead_opt_flag': 0, 'CL0_matrix': [1, 1, 16, 16, 1, 1],
                    'CUB_channel_wise_flag': False, 'CUB_matrix': [1, 1, 16, 16, 1, 1],
                    'batch_bef_group_flag': 0, 'block_dim': [1, 1, 1, 1],
                    'manual_pingpong_buffer': {'AL0_pbuffer': 1, 'AL1_pbuffer': 1, 'AUB_pbuffer': 1,
                    'BL0_pbuffer': 1, 'BL1_pbuffer': 1, 'BUB_pbuffer': 1, 'CL0_pbuffer': 1,
                    'CUB_pbuffer': 1, 'UBG_pbuffer': 1}, 'n_bef_batch_flag': 0, 'n_bef_group_flag': 0,
                    'tbe_compile_para': 0}
    return tiling


def _gen_kernel_name(dedy_shape, w_shape, dx_shape, strides, data_flow):
    dedy_shape_info = "_".join([str(i) for i in dedy_shape])
    w_shape_info = "_".join([str(i) for i in w_shape])
    dx_shape_info = "_".join([str(i) for i in dx_shape])
    stride_shape_info = "_".join([str(i) for i in strides])

    kernel_name = "Conv2DBackpropFilterD_dy_{}_w_{}_x_{}_stride_{}_data_flow_{}".format(
        dedy_shape_info, w_shape_info, dx_shape_info, stride_shape_info, data_flow
    )
    return kernel_name


def _gen_trans_data_case(
    x_dtype,
    dedy_dtype,
    dw_dtype,
    x_shape,
    dedy_shape,
    dw_shape,
    x_format,
    dedy_format,
    dw_format,
    filter_size,
    stride,
    padding,
    dilations=(1, 1, 1, 1),
    groups=1,
    expect="success",
    data_flow="default",
):

    kernel_name = _gen_kernel_name(x_shape, dedy_shape, dw_shape, stride, data_flow)

    fm_list = {
        "shape": util.shape_4d_to_5hd(x_shape, x_dtype, x_format),
        "dtype": x_dtype,
        "format": x_format,
        "ori_shape": x_shape,
        "ori_format": x_format,
    }

    dedy_list = {
        "shape": dedy_shape,
        "dtype": dedy_dtype,
        "format": dedy_format,
        "ori_shape": dedy_shape,
        "ori_format": dedy_format,
    }

    dw_list = {
        "shape": util.shape_4d_to_5hd(dw_shape, dw_dtype, dw_format),
        "dtype": dw_dtype,
        "format": dw_format,
        "ori_shape": dw_shape,
        "ori_format": dw_format,
    }

    filter_sizes = filter_size
    strides = stride
    padding_size = util.gen_padding_size(
        x_shape, dedy_shape, dw_shape, padding, stride, dilations
    )
    data_format = x_format

    if DEBUG_MODE:
        print(
            kernel_name,
            [
                fm_list,
                dedy_list,
                dw_list,
                filter_sizes,
                strides,
                padding,
                dilations,
                groups,
                data_format,
            ],
        )

    return {
        "params": [
            fm_list,
            dedy_list,
            dw_list,
            filter_sizes,
            strides,
            padding_size,
            dilations,
            groups,
            data_format,
        ],
        "case_name": kernel_name,
        "expect": expect,
        "format_expect": [],
        "support_expect": True,
    }


def _test_op_check_supported(test_arg):
    from impl.conv2d_backprop_filter_d import check_supported
    out_backprop = {"ori_shape": (1, 32, 3, 3), "dtype": "float16", "ori_format": "NCHW"}
    y = {"ori_shape": (32, 16, 1, 1), "dtype": "float16", "ori_format": "NCHW"}
    x = {"ori_shape": (1, 16, 5, 5), "dtype": "float16", "ori_format": "NCHW"}
    filter_size = (32, 16, 1, 1)
    check_supported(x,
                    out_backprop,
                    y,
                    filter_size, (1, 1, 2, 2), (0, 0, 0, 0),
                    dilations=(1, 1, 1, 1),
                    groups=1,
                    data_format="NCHW",
                    kernel_name="conv2d_backprop_filter")


def _gen_conv2d_bp_filter_check_support_case():
    ut_case.add_cust_test_func("Ascend910A", test_func=_test_op_check_supported)


def _test_get_op_support_info(test_arg):
    op_info_testcases = conv2d_bp_filter_ut_testcase.op_support_info_testcase
    for testcase in op_info_testcases:
        formatted_case = _gen_trans_data_case(*testcase)
        params = formatted_case["params"]
        params[0]["format"] = "NC1HWC0"
        get_op_support_info(*params)


def _gen_conv2d_bp_filter_op_case():
    
    for test_case in conv2d_bp_filter_ut_testcase.conv2d_bp_filter_op_testcase:
        ut_case.add_case(["Ascend910A"], _gen_trans_data_case(*test_case))
    ut_case.add_cust_test_func(test_func=_test_get_op_support_info)


def _test_nd2nz_format(test_arg):
    fmap_nhwc = tvm.placeholder((1, 7, 7, 16), name="fmap_nhwc", dtype="float16")
    out_nhwc = tvm.placeholder((1, 7, 7, 16), name="out_nhwc", dtype="float16")
    fmap_5hd = trans_data_compute(fmap_nhwc, None, "NHWC", "NC1HWC0")
    out_5hd = trans_data_compute(out_nhwc, None, "NHWC", "NC1HWC0")
    para_dict = {
        "strides": (1, 1),
        "padding": (0, 0, 0, 0),
        "dilations": (1, 1, 1, 1),
        "groups": 1,
        "res_dtype": "float32",
        "kernel_name": "test_nd2nz_format_01"
    }
    filter_fz = tbe.conv2d_backprop_filter(input_x=fmap_5hd,
                                        out_backprop=out_5hd,
                                        filter_sizes=(16, 16, 1, 1),
                                        para_dict=para_dict)
    filter_nhwc = trans_data_compute(filter_fz, {"shape":(16, 1, 16)}, "FRACTAL_Z", "NHWC")
    with tvm.target.cce():
        sch = tbe.auto_schedule(filter_nhwc)
    tensor_list_input = [fmap_nhwc, out_nhwc]
    real_outs = sch.cce_special["real_out_tensor"]
    tensor_list = tensor_list_input + real_outs
    config = {
        "name": "test_nd2nz_format_01",
        "tensor_list": tensor_list
    }
    tbe.build(sch, config)


def _test_nd2nz_format_fp32(test_arg):
    with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=get_soc_mock)):
        with patch("impl.util.platform_adapter.tbe_platform.get_soc_spec", MagicMock(side_effect=get_soc_mock)):
            fmap_nhwc = tvm.placeholder((1, 7, 7, 16), name="fmap_nhwc", dtype="float32")
            out_nhwc = tvm.placeholder((1, 7, 7, 16), name="out_nhwc", dtype="float32")
            fmap_5hd = trans_data_compute(fmap_nhwc, None, "NHWC", "NC1HWC0")
            out_5hd = trans_data_compute(out_nhwc, None, "NHWC", "NC1HWC0")
            para_dict = {
                "strides": (1, 1),
                "padding": (0, 0, 0, 0),
                "dilations": (1, 1, 1, 1),
                "groups": 1,
                "res_dtype": "float32",
                "kernel_name": "test_nd2nz_format_01"
            }
            filter_fz = tbe.conv2d_backprop_filter(input_x=fmap_5hd,
                                                out_backprop=out_5hd,
                                                filter_sizes=(16, 16, 1, 1),
                                                para_dict=para_dict)
            filter_nhwc = trans_data_compute(filter_fz, {"shape":(16, 1, 16)}, "FRACTAL_Z", "NHWC")
            with tvm.target.cce():
                with patch("tbe.common.tiling.tiling_api.get_tiling", MagicMock(side_effect=tiling_mock)):
                    sch = tbe.auto_schedule(filter_nhwc)


def _test_nd2nz_format_bf16(test_arg):
    with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=get_soc_mock)):
        with patch("impl.util.platform_adapter.tbe_platform.get_soc_spec", MagicMock(side_effect=get_soc_mock)):
            fmap_5hd = tvm.placeholder((1, 1, 7, 7, 16), name="fmap_5hd", dtype="bfloat16")
            out_5hd = tvm.placeholder((1, 1, 7, 7, 16), name="out_5hd", dtype="bfloat16")
            para_dict = {
                "strides": (1, 1),
                "padding": (0, 0, 0, 0),
                "dilations": (1, 1, 1, 1),
                "groups": 1,
                "res_dtype": "float32",
                "kernel_name": "test_nd2nz_format_02"
            }
            filter_fz = tbe.conv2d_backprop_filter(input_x=fmap_5hd,
                                                   out_backprop=out_5hd,
                                                   filter_sizes=(16, 16, 1, 1),
                                                   para_dict=para_dict)
            with tvm.target.cce():
                with patch("tbe.common.tiling.tiling_api.get_tiling", MagicMock(side_effect=tiling_mock2)):
                    sch = tbe.auto_schedule(filter_fz)

def _test_nd2nz_format_fp32_c_split(test_arg):
    with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=get_soc_mock)):
        with patch("impl.util.platform_adapter.tbe_platform.get_soc_spec", MagicMock(side_effect=get_soc_mock)):
            fmap_nhwc = tvm.placeholder((1, 7, 7, 16), name="fmap_nhwc", dtype="float32")
            out_nhwc = tvm.placeholder((1, 7, 7, 16), name="out_nhwc", dtype="float32")
            fmap_5hd = trans_data_compute(fmap_nhwc, None, "NHWC", "NC1HWC0")
            out_5hd = trans_data_compute(out_nhwc, None, "NHWC", "NC1HWC0")
            para_dict = {
                "strides": (1, 1),
                "padding": (0, 0, 0, 0),
                "dilations": (1, 1, 1, 1),
                "groups": 1,
                "res_dtype": "float32",
                "kernel_name": "test_nd2nz_format_01"
            }
            filter_fz = tbe.conv2d_backprop_filter(input_x=fmap_5hd,
                                                   out_backprop=out_5hd,
                                                   filter_sizes=(16, 16, 1, 1),
                                                   para_dict=para_dict)
            with tvm.target.cce():
                with patch("tbe.common.tiling.tiling_api.get_tiling", MagicMock(side_effect=tiling_mock)):
                    sch = tbe.auto_schedule(filter_fz)


def _test_nd2nz_format_err1(test_arg):
    try:
        fmap_nhwc = tvm.placeholder((1, 7, 7, 16), name="fmap_nhwc", dtype="float16")
        fmap_5hd = trans_data_compute(fmap_nhwc, None, "NHWCXX", "NC1HWC0")
    except RuntimeError as e:
        print(e)


def _gen_conv2d_bp_filter_nd2nz_format():
    ut_case.add_cust_test_func("Ascend910A", test_func=_test_nd2nz_format)
    ut_case.add_cust_test_func("Ascend910A", test_func=_test_nd2nz_format_fp32)
    ut_case.add_cust_test_func("Ascend910A", test_func=_test_nd2nz_format_err1)
    ut_case.add_cust_test_func("Ascend910A", test_func=_test_nd2nz_format_bf16)
    ut_case.add_cust_test_func("Ascend910A", test_func=_test_nd2nz_format_fp32_c_split)


def _test_conv2d_backprop_filter_compute(test_args):
    def __build_dw_compute(test_case):
        params = test_case["params"]
        with op_context.OpContext():
            with tvm.target.cce():
                fm = tvm.placeholder(params[0]["shape"], name="fmap", dtype="float16", attrs={
                    "ori_shape": params[0]["ori_shape"],
                    "ori_format": params[0]["ori_format"]
                })
                out_backprop = tvm.placeholder(util.shape_4d_to_5hd(params[1]["shape"], "float16", params[1]["ori_format"]),
                                               name="out_backprop", dtype="float16", attrs={
                                                   "ori_shape": params[1]["ori_shape"],
                                                   "ori_format": params[1]["ori_format"]})
                y = params[2]
                filter_size = params[3]
                strides = params[4]
                padding = params[5]
                dilations = params[6]
                groups = 1
                data_format = params[8]
                dedw = conv2d_backprop_filter_compute(fm, out_backprop, y, filter_size, strides, padding,
                                                      dilations, groups, data_format, "conv2d_backprop_filter")
                sch = tbe.auto_schedule(dedw)
                tensor_list = [fm, out_backprop, dedw]
                config = {
                    "name": "conv2d_backprop_filter",
                    "tensor_list": tensor_list
                }
                tbe.build(sch, config)
        
    for test_case in conv2d_bp_filter_ut_testcase.conv2d_bp_filter_compute_testcase:
        formatted_case = _gen_trans_data_case(*test_case)
        formatted_case["params"][5] = test_case[11]
        __build_dw_compute(formatted_case)


_gen_conv2d_bp_filter_op_case()
_gen_conv2d_bp_filter_check_support_case()
_gen_conv2d_bp_filter_nd2nz_format()
ut_case.add_cust_test_func(test_func=_test_conv2d_backprop_filter_compute)


# test mock case
def test_fixpipe_cases(test_args):
    with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=get_soc_mock)):
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=get_soc_mock)):
            test_conv2d_bp_filter_fixpipe_0()
            test_conv2d_bp_filter_fixpipe_1()
            test_conv2d_bp_filter_fixpipe_2()


ut_case.add_cust_test_func(test_func=test_fixpipe_cases)


if __name__ == "__main__":
    ut_case.run("Ascend910A")
    sys.exit(0)
