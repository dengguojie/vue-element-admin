#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from te import tvm
from op_test_frame.ut import OpUT
import conv2D_ut_testcase as tc
from impl.conv2d import conv2d_compute
from impl.conv2d import get_op_support_info

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def dsl_cpu_test_int8():
    fmap = tvm.placeholder((1, 1, 8, 8, 32), name="fmap", dtype="int8", attrs={"ori_shape":(1, 32, 8, 8), "format":"NCHW", "ori_fomat":"NCHW"})
    weight = tvm.placeholder((4, 2, 16, 32), name="weight", dtype="int8", attrs={"ori_shape":(32, 32, 2, 2), "format":"FRACTAL_Z", "ori_fomat":"NCHW"})
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
    fmap = tvm.placeholder((1, 2, 8, 8, 16), name="fmap", dtype="float16", attrs={"ori_shape":(1, 32, 8, 8), "format":"NCHW", "ori_fomat":"NCHW"})
    weight = tvm.placeholder((8, 2, 16, 16), name="weight", dtype="float16", attrs={"ori_shape":(32, 32, 2, 2), "format":"FRACTAL_Z", "ori_fomat":"NCHW"})
    bias_tensor = None
    strides = [1, 1, 1, 1]
    pads = [0, 0, 0, 0]
    dilations = [1, 1, 1, 1]
    conv_res = conv2d_compute(fmap, weight, bias_tensor, None, None, strides, pads, dilations, offset_x=0)
    tensor_list = [fmap, weight, conv_res]
    sch = tvm.create_schedule(conv_res.op)
    fadd = tvm.build(sch, tensor_list, "c", "llvm", name="fadd")
    ctx = tvm.cpu(0)
    
def gen_kernel_name(input_shape, weights_shape):
    dedy_shape_info = '_'.join([str(i) for i in input_shape])
    w_shape_info = '_'.join([str(i) for i in weights_shape])

    kernel_name = 'conv2d_x_{}_w_{}'.format(
        dedy_shape_info, w_shape_info)
    return kernel_name

def gen_trans_data_case(inputs, weights, bias, offset_w, outputs, strides, pads, dilations, expect):

    input_shape = inputs.get('ori_shape')
    weights_shape = weights.get('ori_shape')
    kernel_name = gen_kernel_name(input_shape, weights_shape)
    return {"params": [inputs, weights, bias, offset_w, outputs, strides, pads, dilations],
            "case_name": kernel_name,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

def _test_get_op_support_info(test_arg):
    for test_case  in tc.op_support_info_testcase:
        formatted_case = gen_trans_data_case(*test_case[1:])
        params = formatted_case["params"]
        get_op_support_info(*params)

print("adding Conv2D op testcases")
for test_case  in tc.conv2D_ut_testcase:
    ut_case.add_case(test_case[0], gen_trans_data_case(*test_case[1:]))

print("adding Connv2D op support info of slice testcase")
ut_case.add_cust_test_func(test_func=_test_get_op_support_info)

def test_op_check_supported_1(test_arg):
    from impl.conv2d import check_supported
    input_list = [
        {
            'shape': (64, 1, 100000, 4096, 16),
            'ori_shape': (64, 3, 100000, 4096),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (33, 3, 255, 255),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'ori_format': 'NCHW',
            'dtype': 'float16'
        }, (1, 1, 63, 63), (255, 255, 255, 255), (1, 1, 255, 255), 1, 'NCHW', 127, 'conv2d_check_support']
    check_supported(*input_list)

print("adding conv2d test_op_check_supported_1 testcase")
ut_case.add_cust_test_func(test_func=test_op_check_supported_1)

if __name__ == '__main__':
    ut_case.run(["Ascend910", "Ascend310"])
    dsl_cpu_test_int8()
    dsl_cpu_test_fp16()
    exit(0)
