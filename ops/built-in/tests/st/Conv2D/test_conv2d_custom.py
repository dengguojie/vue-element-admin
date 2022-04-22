#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''

from impl.dynamic.conv2d import conv2d_generalization
from te import tvm
from impl.conv2d import conv2d_compute


def test_conv2d_fuzzbuild_generalization():
    input_list = [
        {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', 0, 'conv2d_fuzz_build_generalization']
    conv2d_generalization(*input_list)

def test_conv2d_fuzzbuild_generalization_01():
    input_list = [
        {
            'shape': (16, 1, -1, -1, 16),
            'ori_shape': (16, 3, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            "ori_range": [[16,16], [3,3], [16,31], [16,31]]
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (-1, -1, -1, -1), (1, 1, 1, 1), 1, 'NCHW', 0, 'conv2d_fuzz_build_generalization']
    conv2d_generalization(*input_list)


def dsl_cpu_test_int8():
    fmap = tvm.placeholder((1, 1, 8, 8, 32), name="fmap", dtype="int8", attrs={"ori_shape":(1, 32, 8, 8), "format":"NCHW", "ori_format":"NCHW"})
    weight = tvm.placeholder((4, 2, 16, 32), name="weight", dtype="int8", attrs={"ori_shape":(32, 32, 2, 2), "format":"FRACTAL_Z", "ori_format":"NCHW"})
    bias_tensor = None
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]
    conv_res = conv2d_compute(fmap, weight, bias_tensor, None, None, strides, pads, dilations, offset_x=0)
    tensor_list = [fmap, weight, conv_res]
    sch = tvm.create_schedule(conv_res.op)
    fadd = tvm.build(sch, tensor_list, "c", "llvm", name="fadd")
    ctx = tvm.cpu(0)


def dsl_cpu_test_fp16():
    fmap = tvm.placeholder((1, 2, 8, 8, 16), name="fmap", dtype="float16", attrs={"ori_shape":(1, 32, 8, 8), "format":"NCHW", "ori_format":"NCHW"})
    weight = tvm.placeholder((8, 2, 16, 16), name="weight", dtype="float16", attrs={"ori_shape":(32, 32, 2, 2), "format":"FRACTAL_Z", "ori_format":"NCHW"})
    bias_tensor = None
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]
    conv_res = conv2d_compute(fmap, weight, bias_tensor, None, None, strides, pads, dilations, offset_x=0)
    tensor_list = [fmap, weight, conv_res]
    sch = tvm.create_schedule(conv_res.op)
    fadd = tvm.build(sch, tensor_list, "c", "llvm", name="fadd")
    ctx = tvm.cpu(0)

if __name__ == "__main__":
    test_conv2d_fuzzbuild_generalization()
    test_conv2d_fuzzbuild_generalization_01()
    dsl_cpu_test_int8()
    dsl_cpu_test_fp16()
    exit(0)
