#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def binary_dma_test(test_arg):
    from tbe import tvm
    from tbe.dsl import auto_schedule
    from tbe.dsl.base import operation
    from impl.dynamic.conv2d import _conv2d_compute
    inputs = tvm.placeholder(
        (operation.var("batch_n", [1, 1]),
         1,
         operation.var("fmap_h", [5, 5]),
         operation.var("fmap_w", [5, 5]),
         16),
    name="fmap", dtype="float16",
    attrs={"ori_shape": (-1, 16, -1, -1),
           "ori_format": "NCHW",
           "dtype": "float16",
           "range": [(1, 1), (1, 1), (5, 5), (5, 5), (16, 16)],
           "format": "NC1HWC0"})

    weights = tvm.placeholder(
        (9, 1, 16, 16),
        name="weight",
        dtype="float16",
        attrs={"ori_shape": [1, 16, 3, 3],
               "dtype": "float16",
               "ori_format": "NCHW",
               "range": [(1, 1), (16, 16), (3, 3), (3, 3)], "format": "FRACTAL_Z"})

    option_dict = {"res_dtype": "float16",
                   "optim_dict": {"c0_optim_flg": False, "use_v200_c04_flg": False,
                                  "v220_c04_mode": "disabled", "invalid_data_rm": False},
                   "fusion_para": {"input_memory_type": 0, "output_memory_type": 0,
                                   "valid_shape": (), "slice_offset": (), "l1_fusion_type": -1},
                   "ori_shape": [0, 0, 0, 0], "dma_flag": True}

    bias = None
    offset_w = None
    outputs = {"ori_shape": (-1, 16, -1, -1), "ori_format": "NCHW", "dtype": "float16"}
    strides = (-1, -1, -1, -1)
    pads = (1, 1, 1, 1)
    dilations = (1, 1, 1, 1)
    groups = 1
    data_format = "NCHW"
    offset_x = 0
    kernel_name = "conv2d_binary_dma"
    with operation.dynamic():
        with operation.ComputeContext():
            res = _conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                groups, data_format, offset_x, kernel_name, False, option_dict)
            conv_out = res['op_res'][0]
            out = [conv_out]
            with tvm.target.cce():
                sch = auto_schedule(res)

print("adding Conv2D v200 int4 ut testcases")
ut_case.add_cust_test_func(["Ascend910A"], test_func=binary_dma_test)