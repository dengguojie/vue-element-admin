#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.fix_pipe import fixpipe_compute
from te import tvm
from tbe.dsl import auto_schedule
from unittest.mock import MagicMock
from unittest.mock import patch
from impl.conv2d_backprop_filter_d import conv2d_backprop_filter_compute
from impl.deconvolution import deconvolution_compute
from impl.mat_mul import mat_mul_compute
from te.tvm.target import cce


ut_case = OpUT("FixPipe", "impl.fix_pipe", "fix_pipe")

vals = {
    ("CORE_NUM", ): 48,
    ("CUBE_VECTOR_SPLIT",): True,
    ("UB_SIZE", ): 196608,
    ("L0A_SIZE", ): 65536,
    ("L0B_SIZE", ): 65536,
    ("L1_SIZE", ): 524288,
    ("L0C_SIZE", ): 131072,
    ("Compiler_arch",): "dav-c220-cube",
    ("AICORE_TYPE",): "AiCore",
    ("SOC_VERSION",): "Ascend920A",
    ("Intrinsic_fix_pipe_unit_list",): True,
    ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): True,
    ("Intrinsic_fix_pipe_l0c2ub",) : True,
    ("Intrinsic_fix_pipe_l0c2out",) : True,
    ("Intrinsic_data_move_l0c2ub",) : True,
    ("Intrinsic_data_move_l12bt",) : True,
    ("Intrinsic_data_move_ub2l1",) : True,
    ("Intrinsic_mmad", "f162f32",) : True,
    ("CUBE_VECTOR_SPLIT",) : True,
}

def get_soc_mock(*args):
    return vals[args]

def test_conv2d_bp_filter_fixpipe_0():
    with cce():
        fmap_tensor = tvm.placeholder((1, 2, 28, 28, 16), name="fmap", dtype="float16", attrs={"ori_shape": (1, 32, 28, 28), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedy_tensor = tvm.placeholder((1, 1, 26, 26, 16), name="dedy", dtype="float16", attrs={"ori_shape": (1, 16, 26, 26), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedw = {"shape": (16, 32, 3, 3), "dtype": "float32", "ori_shape": (16, 32, 3, 3), "format": "NCHW", "ori_format": "NCHW"}
        filter_size = (16, 32, 3, 3)
        strides = (1, 1, 1, 1)
        pads = (0, 0, 0, 0)
        dilations=(1, 1, 1, 1)
        groups = 1
        data_format = "NCHW"
        dedw_tensor = conv2d_backprop_filter_compute(fmap_tensor, dedy_tensor, dedw, filter_size, strides, pads, dilations, groups, data_format)
        output_dict = {"shape": (16, 3, 3, 32), "dtype": "float32", "format": "NHWC"}
        res = fixpipe_compute(dedw_tensor, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
        tensor_list = [fmap_tensor, dedy_tensor, res]
        # sch = auto_schedule(res)


def test_conv2d_bp_filter_fixpipe_1():
    with cce():
        fmap_tensor = tvm.placeholder((1, 4, 28, 28, 8), name="fmap", dtype="float32", attrs={"ori_shape": (1, 32, 28, 28), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedy_tensor = tvm.placeholder((1, 2, 26, 26, 8), name="dedy", dtype="float32", attrs={"ori_shape": (1, 16, 26, 26), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedw = {"shape": (16, 32, 3, 3), "dtype": "float32", "ori_shape": (16, 32, 3, 3), "format": "NCHW", "ori_format": "NCHW"}
        filter_size = (16, 32, 3, 3)
        strides = (1, 1, 1, 1)
        pads = (0, 0, 0, 0)
        dilations=(1, 1, 1, 1)
        groups = 1
        data_format = "NCHW"
        dedw_tensor = conv2d_backprop_filter_compute(fmap_tensor, dedy_tensor, dedw, filter_size, strides, pads, dilations, groups, data_format)
        output_dict = {"shape": (16, 3, 3, 32), "dtype": "float32", "format": "NHWC"}
        res = fixpipe_compute(dedw_tensor, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
        tensor_list = [fmap_tensor, dedy_tensor, res]
        # sch = auto_schedule(res)


def test_conv2d_bp_filter_fixpipe_2():
    with cce():
        fmap_tensor = tvm.placeholder((1, 2, 28, 28, 16), name="fmap", dtype="float16", attrs={"ori_shape": (1, 32, 28, 28), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedy_tensor = tvm.placeholder((1, 1, 26, 26, 16), name="dedy", dtype="float16", attrs={"ori_shape": (1, 16, 26, 26), "format": "NC1HWC0", "ori_format": "NCHW"})
        dedw = {"shape": (16, 32, 3, 3), "dtype": "float32", "ori_shape": (16, 32, 3, 3), "format": "NCHW", "ori_format": "NCHW"}
        filter_size = (16, 32, 3, 3)
        strides = (1, 1, 1, 1)
        pads = (0, 0, 0, 0)
        dilations=(1, 1, 1, 1)
        groups = 1
        data_format = "NCHW"
        dedw_tensor = conv2d_backprop_filter_compute(fmap_tensor, dedy_tensor, dedw, filter_size, strides, pads, dilations, groups, data_format)
        output_dict = {"shape": (18, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_Z"}
        res = fixpipe_compute(dedw_tensor, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
        tensor_list = [fmap_tensor, dedy_tensor, res]
        # sch = auto_schedule(res)

def test_conv2d_bp_filter_fixpipe_3():
    try:
        with cce():
            fmap_tensor = tvm.placeholder((1, 2, 28, 28, 16), name="fmap", dtype="float16", attrs={"ori_shape": (1, 32, 28, 28), "format": "NC1HWC0", "ori_format": "NCHW"})
            dedy_tensor = tvm.placeholder((1, 1, 26, 26, 16), name="dedy", dtype="float16", attrs={"ori_shape": (1, 16, 26, 26), "format": "NC1HWC0", "ori_format": "NCHW"})
            dedw = {"shape": (16, 32, 3, 3), "dtype": "float32", "ori_shape": (16, 32, 3, 3), "format": "NCHW", "ori_format": "NCHW"}
            filter_size = (16, 32, 3, 3)
            strides = (1, 1, 1, 1)
            pads = (0, 0, 0, 0)
            dilations=(1, 1, 1, 1)
            groups = 1
            data_format = "NCHW"
            dedw_tensor = conv2d_backprop_filter_compute(fmap_tensor, dedy_tensor, dedw, filter_size, strides, pads, dilations, groups, data_format)
            output_dict = {"shape": (18, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ"}
            res = fixpipe_compute(dedw_tensor, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
            tensor_list = [fmap_tensor, dedy_tensor, res]
            # sch = auto_schedule(res)
    except RuntimeError as e:
        print("fixpipe test mock")

def fixpie_deconv_case0():
    filter_frac = (18, 2, 16, 16)
    out_shape_5hd = (16, 2, 2, 2, 16)
    input_size = (16, 32, 4, 4)
    input_size_5hd = (16, 2, 4, 4, 16)
    data_type = "float16"
    try:
        with cce():
            weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                    attrs={"ori_shape": (32, 32, 3, 3), "dtype":data_type, "ori_format": "NCHW"})
            dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                attrs={"ori_shape": (16, 32, 2, 2), "dtype":data_type, "ori_format": "NCHW"})
            eltwise_plh = tvm.placeholder((input_size[0], (input_size[1] + 15) // 16, input_size[2] * input_size[3], 16),
                                            name="fixpipe_eltwise", dtype="float16",
                                            attrs={"ori_format": "NCHW", "ori_shape": input_size})
            y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW", "shape": input_size_5hd,
                 "format": "NC1HWC0"}
            out = deconvolution_compute(dedy, weight, None, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                    dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                    kernel_name="deconvolution")
            res = fixpipe_compute(out, eltwise_plh, None, None, None, None, None, None, None, None, y, [], [], "")
            sch = auto_schedule(res)
    except RuntimeError as e:
        print("deconv fixpipe base exception.")
        print("exception:", e)

def fixpipe_deconv_channel_split():
    filter_frac = (36, 2, 16, 8)
    out_shape_5hd = (16, 4, 2, 2, 8)
    input_size = (16, 32, 4, 4)
    input_size_5hd = (16, 4, 4, 4, 8)
    data_type = "float32"
    try:
        with cce():
            weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                    attrs={"ori_shape": (32, 32, 3, 3), "dtype":data_type, "ori_format": "NCHW"})
            dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                attrs={"ori_shape": (16, 32, 2, 2), "dtype":data_type, "ori_format": "NCHW"})
            y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW", "shape": input_size_5hd,
                    "format": "NC1HWC0"}
            out = deconvolution_compute(dedy, weight, None, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                    dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                    kernel_name="deconvolution")
            res = fixpipe_compute(out, None, None, None, None, None, None, None, None, None, y, [], [], "")
            sch = auto_schedule(res)
    except RuntimeError as e:
        print("deconv fixpipe channel_split exception.")
        print("exception:", e)


def fixpipe_deconv_shape_error():
    filter_frac = (18, 2, 16, 16)
    out_shape_5hd = (16, 2, 2, 2, 16)
    input_size = (16, 32, 4, 4)
    input_size_5hd = (16, 2, 4, 4, 16)
    data_type = "float16"
    try:
        with cce():
            weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                    attrs={"ori_shape": (32, 32, 3, 3), "dtype":data_type, "ori_format": "NCHW"})
            dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                attrs={"ori_shape": (16, 32, 2, 2), "dtype":data_type, "ori_format": "NCHW"})
            eltwise_plh = tvm.placeholder((input_size[0], (input_size[1] + 15) // 16, input_size[2] * input_size[3], 16),
                                            name="fixpipe_eltwise", dtype="float16",
                                            attrs={"ori_format": "NCHW", "ori_shape": input_size})
            y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW", "shape": input_size_5hd,
                    "format": "NC1HWC0"}
            out = deconvolution_compute(dedy, weight, None, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                    dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                    kernel_name="deconvolution")
            y["shape"] = [1,1]
            res = fixpipe_compute(out, eltwise_plh, None, None, None, None, None, None, None, None, y, [], [], "")
            sch = auto_schedule(res)
    except RuntimeError as e:
        print("deconv fixpipe shape_error exception.")
        print("exception:", e)


def test_matmul_fixpipe_op_name():
    with cce():
        x1 = tvm.placeholder((4, 2, 16, 16), name="tensor_a", dtype="float16", attrs={"ori_shape": (32, 64), "format": "FRACTAL_NZ", "ori_format": "ND"})
        x2 = tvm.placeholder((2, 4, 16, 16), name="tensor_b", dtype="float16", attrs={"ori_shape": (64, 32), "format": "FRACTAL_NZ", "ori_format": "ND"})
        bias = tvm.placeholder((32,), name="tensor_bias", dtype="float32", attrs={"format": "ND", "ori_format": "ND", "ori_shape": (32,)})
        output_y = {"shape": (2, 2, 16, 16), "dtype": "float16", "ori_shape": (32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(x1, x2, bias, None, output_y, False, False, 0)
        y = {"shape": (2, 2, 16, 16), "dtype": "float16", "ori_shape": (32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
        res = fixpipe_compute(matmul_out, None, None, None, None, None, None, None, None, None, y, [], [], "")
        tensor_list = [x1, x2, bias, res]
        sch = auto_schedule(res)
        assert res.op.name != res.op.input_tensors[0].op.name

def test_fixpipe_cases(test_args):
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=get_soc_mock)):
        with patch("impl.util.platform_adapter.tbe_platform.get_soc_spec", MagicMock(side_effect=get_soc_mock)):
            with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=get_soc_mock)):
                with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=get_soc_mock)):
                    test_conv2d_bp_filter_fixpipe_0()
                    test_conv2d_bp_filter_fixpipe_1()
                    test_conv2d_bp_filter_fixpipe_2()
                    test_conv2d_bp_filter_fixpipe_3()
                    fixpie_deconv_case0()
                    fixpipe_deconv_channel_split()
                    fixpipe_deconv_shape_error()
                    test_matmul_fixpipe_op_name()


ut_case.add_cust_test_func(test_func=test_fixpipe_cases)


if __name__ == '__main__':
    # ut_case.run("Ascend310")
    ut_case.run()
