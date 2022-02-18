#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import conv2D_ut_testcase as tc
from te import tvm
from impl.util.util_conv2d_dynamic import Conv2dParaProcess

ut_case = OpUT("Conv2D", "impl.dynamic.conv2d", "conv2d")

def test_conv2d_param_process(test_arg):
    fmap = tvm.placeholder((-1, 2, -1, -1, 16), name="fmap", dtype="float16", attrs={"ori_shape": (-1, 32, -1, -1), "format": "NCHW", "ori_format": "NCHW", "range": [(1, 2), (32, 32), (8, 16), (8, 16)]})
    weight = tvm.placeholder((8, 2, 16, 16), name="weight", dtype="float16", attrs={"ori_shape": (32, 32, 2, 2), "format": "FRACTAL_Z", "ori_format": "NCHW"})
    bias_tensor = None
    offset_w_tensor = None
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]
    outputs = {"ori_shape": [-1, 128, -1, -1], "ori_format": "NCHW", "dtype": "float16"}

    ori_paras = {
        "inputs": fmap, "weights": weight, "bias": bias_tensor, "offset_w": offset_w_tensor,
        "outputs": outputs, "strides": strides, "pads": pads, "dilations": dilations,
        "groups": 1, "data_format": "NCHW", "kernel_name": "conv2d", "optim_dict": {}
    }
    conv_para = Conv2dParaProcess(ori_paras)
    conv_para.config_paras()

print("adding Connv2D dyanmic op param process")
ut_case.add_cust_test_func(test_func=test_conv2d_param_process)


if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710", "Ascend310"])