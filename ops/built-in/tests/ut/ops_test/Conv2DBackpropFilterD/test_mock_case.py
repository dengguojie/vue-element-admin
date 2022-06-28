#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from te import tvm
from te.tvm.target import cce
from tbe.dsl import auto_schedule

from impl.fix_pipe import fixpipe_compute
from impl.fix_pipe import fix_pipe
from impl.conv2d_backprop_filter_d import conv2d_backprop_filter_compute



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


def test_milan_dma_copy_case():
    fmap_5hd = tvm.placeholder((5, 1, 97,4097, 16), name="fmap", dtype="float16", attrs={"ori_shape": (5, 7, 97,4097), "format": "NC1HWC0", "ori_format": "NCHW"})
    out_5hd = tvm.placeholder((5,1, 1, 24, 16), name="out", dtype="float16", attrs={"ori_shape": (5, 2, 1, 24), "format": "NC1HWC0", "ori_format": "NCHW"})
    dedw = {"shape": (14136,1,16,16), "dtype": "float32", "ori_shape": ((2,7,38,372)), "format": "FRACTAL_Z", "ori_format": "NCHW"}
    filter_fz =conv2d_backprop_filter_compute(fmap_5hd, out_5hd, dedw, (2,7,38,372), (230, 160), (0, 0, 0, 0),
                                              (1, 1, 1, 1), 1, "NCHW")
    with tvm.target.cce():
        sch = auto_schedule(filter_fz)
