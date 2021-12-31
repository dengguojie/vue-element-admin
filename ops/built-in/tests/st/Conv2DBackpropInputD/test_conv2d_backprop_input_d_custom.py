#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
the Conv2DBackpropInputD test
"""
import sys
from math import ceil as math_ceil
from unittest.mock import MagicMock
from unittest.mock import patch

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


def _test_conv2d_bp_input_fp32_case_1():
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

def _test_conv2d_bp_input_fp32_case_2():
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

def _test_conv2d_bp_input_opti_fp16_case_1():
   with patch("tbe.common.platform.intrinsic_check_support", MagicMock(side_effect=check_intrinsic_cube_vector_split)):
        with patch("impl.util.platform_adapter.tbe_platform.get_soc_spec", MagicMock(side_effect=side_effects)):
            with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
                with cce():
                    x = tvm.placeholder((2, 7, 7, 2048), name="x", dtype="float16",
                                        attrs={
                                            "ori_shape": (2, 7, 7, 2048), "format": "NHWC", 
                                            "ori_format": "NHWC"
                                        })
                    x_5hd = trans_data_compute(x, None, "NHWC", "NC1HWC0")
                    weight = tvm.placeholder((32, 128, 16, 16), name="filter", dtype="float16",
                                                attrs={
                                                    "ori_shape": (1, 1, 512, 2048),
                                                    "format": "FRACTAL_Z",
                                                    "ori_format": "HWCN"
                                                })
                    y = {"shape": (2, 128, 7, 7, 16), "ori_shape": (2, 7, 7, 512), "format": "NC1HWC0", "ori_format": "NHWC", "dtype": "float16"}
                    dx = conv2d_backprop_input_d_compute(weight, x_5hd, y, (2, 7, 7, 512), (1, 1), "VALID", (1, 1, 1, 1))
                    trans_out = {"shape": (2, 7, 7, 512), "ori_shape": (2, 7, 7, 512), "format": "NHWC", "ori_format": "NHWC", "dtype": "float16"}
                    out = trans_data_compute(dx, trans_out, "NC1HWC0", "NHWC")
                    sch = auto_schedule(out)


if __name__ == "__main__":
    _test_conv2d_bp_input_fp32_case_1()
    _test_conv2d_bp_input_fp32_case_2()
    _test_conv2d_bp_input_opti_fp16_case_1()