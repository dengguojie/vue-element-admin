#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.fix_pipe import fixpipe_compute
from impl.fixpipe_op.fixpipe_factory import FixpipeFactory
from impl.fixpipe_op.fixpipe_conv2d_backprop_filter import FixpipeConv2dBackpropFilter
from te import tvm
from tbe.dsl import auto_schedule
from unittest.mock import MagicMock
from unittest.mock import patch
from impl.conv2d_backprop_filter_d import conv2d_backprop_filter_compute
from te.tvm.target import cce



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
        fixpipe = FixpipeConv2dBackpropFilter("conv2d_backprop_filter", dedw_tensor, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
        res = fixpipe.fixpipe_compute()
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
        fixpipe = FixpipeConv2dBackpropFilter("conv2d_backprop_filter", dedw_tensor, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
        res = fixpipe.fixpipe_compute()
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
        fixpipe = FixpipeConv2dBackpropFilter("conv2d_backprop_filter", dedw_tensor, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
        res = fixpipe.fixpipe_compute()
        tensor_list = [fmap_tensor, dedy_tensor, res]
        # sch = auto_schedule(res)



def test_fixpipe_cases():
    with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=get_soc_mock)):
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=get_soc_mock)):
            test_conv2d_bp_filter_fixpipe_0()
            test_conv2d_bp_filter_fixpipe_1()
            test_conv2d_bp_filter_fixpipe_2()


if __name__ == '__main__':
    test_fixpipe_cases()
