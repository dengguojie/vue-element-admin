#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from unittest.mock import MagicMock
from unittest.mock import patch
from impl.deconvolution import deconvolution_compute
from impl.deconvolution import get_op_support_info
from op_test_frame.ut import OpUT
from te import tvm

from te.tvm.target import cce
from tbe.dsl import auto_schedule

ut_case = OpUT("Deconvolution", "impl.deconvolution", "deconvolution")

"""
the deconvolution test
"""

support_intrinsic_no_fixpipe = {
    ("Intrinsic_fix_pipe_l0c2ub",) : False,
    ("Intrinsic_fix_pipe_l0c2out",) : False,
    ("Intrinsic_data_move_l0c2ub",) : True,
    ("Intrinsic_data_move_l12bt",) : True,
    ("Intrinsic_data_move_ub2l1",) : True,
    ("Intrinsic_mmad", "f162f32",) : False,
    ("CUBE_VECTOR_SPLIT",) : False,
}

support_intrinsic_mix = {
    ("Intrinsic_fix_pipe_l0c2ub",) : True,
    ("Intrinsic_fix_pipe_l0c2out",) : True,
    ("Intrinsic_data_move_l0c2ub",) : True,
    ("Intrinsic_data_move_l12bt",) : True,
    ("Intrinsic_data_move_ub2l1",) : True,
    ("Intrinsic_mmad", "f162f32",) : True,
    ("CUBE_VECTOR_SPLIT",) : True,
}

def check_intrinsic_mock_mix(*args):
    return support_intrinsic_mix[args]

def check_intrinsic_mock_no_fixpipe(*args):
    return support_intrinsic_no_fixpipe[args]



def test_op_group_requant():
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_no_fixpipe)):
        x = {"ori_shape": (1, 192, 48, 80), "dtype": "int8", "ori_format": "NCHW", "shape": (1, 6, 48, 80, 32), "format":"NC1HWC0"}
        weight = {"ori_shape": (144, 64, 3, 3), "dtype": "int8", "ori_format": "NCHW", "shape": (54, 3, 16, 32), "format": "FRACTAL_NZ"}
        bias = None
        y = {"ori_shape": (1, 144, 48, 80), "dtype": "int8", "ori_format": "NCHW",  "shape": (1, 5, 48, 80, 32), "format":"NC1HWC0"}
        from impl.deconvolution import deconvolution
        try:
            deconvolution(x, weight, bias, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                        dilations=(1, 1, 1, 1), groups=10, data_format="NCHW", offset_x=0,
                        kernel_name="deconvolution")
        except RuntimeError as e:
            print(e)
            pass

def test_op_mix_case1_mock():
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        filter_frac = (144, 16, 16, 16)
        out_shape_5hd = (2, 16, 14, 14, 16)
        input_size = (2, 256, 28, 28)
        data_type = "float16"
        with cce():
            weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                    attrs={"ori_shape": (256, 256, 3, 3), "dtype":data_type, "ori_format": "NCHW"})
            dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                attrs={"ori_shape": (2, 256, 14, 14), "dtype":data_type, "ori_format": "NCHW"})
            y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW"}
            out = deconvolution_compute(dedy, weight, None, None, y, (1, 1, 2, 2), (0, 1, 0, 1),
                    dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                    kernel_name="deconvolution")
            sch = auto_schedule(out)

def test_op_mix_case2_mock():
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        filter_frac = (2, 2, 16, 16)
        out_shape_5hd = (2, 2, 7, 7, 16)
        input_size = (2, 32, 14, 14)
        data_type = "float16"
        with cce():
            weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                    attrs={"ori_shape": (32, 32, 1, 1), "dtype":data_type, "ori_format": "NCHW"})
            dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                attrs={"ori_shape": (2, 32, 7, 7), "dtype":data_type, "ori_format": "NCHW"})
            y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW"}
            out = deconvolution_compute(dedy, weight, None, None, y, (1, 1, 2, 2), (0, 0, 0, 0),
                        dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                        kernel_name="deconvolution")
            sch = auto_schedule(out)

def test_op_mix_case3_mock():
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        filter_frac = (2, 2, 16, 16)
        out_shape_5hd = (2, 2, 7, 7, 16)
        input_size = (2, 32, 7, 7)
        data_type = "float16"
        with cce():
            weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                    attrs={"ori_shape": (32, 32, 1, 1), "dtype":data_type, "ori_format": "NCHW"})
            dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                attrs={"ori_shape": (2, 32, 7, 7), "dtype":data_type, "ori_format": "NCHW"})
            bias = tvm.placeholder((32,), name="bias", dtype="float32",
                                attrs={"ori_shape": (32,), "dtype":data_type, "ori_format": "ND"})
            y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW"}
            out = deconvolution_compute(dedy, weight, bias, None, y, (1, 1, 1, 1), (0, 0, 0, 0),
                        dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                        kernel_name="deconvolution")
            sch = auto_schedule(out)

def test_op_mix_case4_mock():
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_mix)):
        filter_frac = (144, 16, 16, 16)
        out_shape_5hd = (2, 16, 14, 14, 16)
        input_size = (2, 256, 28, 28)
        data_type = "float16"
        with cce():
            weight = tvm.placeholder(filter_frac, name="filter", dtype=data_type,
                                    attrs={"ori_shape": (256, 256, 3, 3), "dtype":data_type, "ori_format": "NCHW"})
            dedy = tvm.placeholder(out_shape_5hd, name="dedy", dtype=data_type,
                                attrs={"ori_shape": (2, 256, 14, 14), "dtype":data_type, "ori_format": "NCHW"})
            bias = tvm.placeholder((32,), name="bias", dtype="float32",
                                attrs={"ori_shape": (256,), "dtype":data_type, "ori_format": "ND"})
            y = {"ori_shape" : input_size, "dtype" : data_type, "ori_format" : "NCHW"}
            out = deconvolution_compute(dedy, weight, bias, None, y, (1, 1, 2, 2), (0, 1, 0, 1),
                    dilations=(1, 1, 1, 1), groups=1, data_format="NCHW", offset_x=0,
                    kernel_name="deconvolution")
            sch = auto_schedule(out)


def test_op_compute_int8():
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_no_fixpipe)):
        x = tvm.placeholder(
            (1, 6, 48, 80, 32),
            name="x",
            dtype="int8",
            attrs={"ori_shape": (1, 192, 48, 80), "ori_format": "NCHW"},
        )
        weight_tensor = tvm.placeholder(
            (54, 3, 16, 32),
            name="filter",
            dtype="int8",
            attrs={
                "ori_shape": (144, 64, 3, 3),
                "ori_format": "NCHW",
            },
        )
        dedx_list = {
            "ori_shape": (1,144,48,80),
            "shape": (1,5,48,80,32),
            "dtype": "int8",
            "ori_format": "NCHW",
            "format": "NC1HWC0",
        }
        try:
            deconvolution_compute(
            x, weight_tensor, None, None, dedx_list,
            (1, 1, 1, 1), (0, 0, 0, 0),
            dilations=(1, 1, 1, 1), groups=10,
            )
        except RuntimeError as e:
            print(e)
            pass


def test_split_deconvolution():
    with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_mock_no_fixpipe)):
        x = {"ori_shape": (1, 192, 48, 80), "dtype": "int8", "ori_format": "NCHW", "shape": (1, 6, 48, 80, 32), "format":"NC1HWC0"}
        weight = {"ori_shape": (144, 64, 3, 3), "dtype": "int8", "ori_format": "NCHW", "shape": (54, 3, 16, 32), "format": "FRACTAL_NZ"}
        bias = None
        y = {"ori_shape": (1, 144, 48, 80), "dtype": "int8", "ori_format": "NCHW",  "shape": (1, 5, 48, 80, 32), "format":"NC1HWC0"}
        get_op_support_info(x, weight, bias, None, y, (1, 1), (0, 0, 0, 0))


if __name__ == "__main__":
    test_op_group_requant()
    test_op_mix_case1_mock()
    test_op_mix_case2_mock()
    test_op_mix_case3_mock()
    test_op_mix_case4_mock()
    test_op_compute_int8()
    test_split_deconvolution()