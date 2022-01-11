#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from te import tvm
from op_test_frame.ut import OpUT
import conv2D_ut_testcase as tc
from impl.conv2d import conv2d_compute
from impl.conv2d import get_op_support_info
from tbe.dsl.compute.conv_compute import conv

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

    kernel_name = 'conv2d_x_{}_w_{}'.format(dedy_shape_info, w_shape_info)
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
for test_case in tc.conv2D_ut_testcase:
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

def test_conv2d_fuzz_build_generalization(test_arg):
    from impl.dynamic.conv2d import conv2d_generalization
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
print("adding conv2d test_conv2d_fuzz_build_generalization testcase")
ut_case.add_cust_test_func(test_func=test_conv2d_fuzz_build_generalization)

def test_conv2d_fuzz_build_generalization_01(test_arg):
    from impl.dynamic.conv2d import conv2d_generalization
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
print("adding conv2d test_conv2d_fuzz_build_generalization_01 testcase")
ut_case.add_cust_test_func(test_func=test_conv2d_fuzz_build_generalization_01)

def test_conv2d_fuzz_build_tilingcase(test_arg):
    import json
    from impl.dynamic.conv2d import conv2d
    from tbe.common.context import get_context
    from tbe.common.context import op_context
    with op_context.OpContext("dynamic"):
        get_context().set_build_type("fuzzily_build")
        get_context().add_addition("max_kernel_id", -1)
        missing_info = [{
                            "inputs": [{
                                "index": 0,
                                "tensor": [{
                                    "range": [
                                        [16, 32],
                                        [3, 3],
                                        [256, 512],
                                        [256, 512]
                                    ],
                                    "shape": [-1, 3, -1, -1]
                                }]
                            }]
                        }, {
                            "inputs": [{
                                "index": 0,
                                "tensor": [{
                                    "range": [
                                        [16, 32],
                                        [3, 3],
                                        [512, 1024],
                                        [512, 1024]
                                    ],
                                    "shape": [-1, 3, -1, -1]
                                }]
                            }]
                        }]
        get_context().add_addition("missing_support_info", json.dumps(missing_info))
        input_list = [
            {
                'shape': (-1, 1, -1, -1, 16),
                'ori_shape': (-1, 3, -1, -1),
                'ori_format': 'NCHW',
                'format': 'NC1HWC0',
                'dtype': 'float16',
                'range': ((16, 32), (3, 3), (256, 1024), (256, 1024))
            }, {
                'ori_shape': (33, 3, 3, 5),
                'ori_format': 'NCHW',
                'format': 'FRACTAL_Z',
                'dtype': 'float16'
            }, None, None, {
                'shape': (-1, 3, -1, -1, 16),
                'ori_shape': (-1, 33, -1, -1),
                'ori_format': 'NCHW',
                'format': 'NC1HWC0',
                'dtype': 'float16'
            }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', 0, 'test_conv2d_fuzz_build_tilingcase']
        conv2d(*input_list)
# print("adding conv2d test_conv2d_fuzz_build_tilingcase testcase")
# ut_case.add_cust_test_func(test_func=test_conv2d_fuzz_build_tilingcase)

def test_conv2d_fuzz_build_correct_range(test_arg):
    import json
    from impl.dynamic.conv2d import conv2d
    from impl.dynamic.conv2d import conv2d_generalization
    from tbe.common.context import get_context
    from tbe.common.context import op_context
    with op_context.OpContext("dynamic"):
        get_context().set_build_type("fuzzily_build")
        get_context().add_addition("max_kernel_id", -1)
        missing_info = [{
                            "inputs": [{
                                "index": 0,
                                "tensor": [{
                                    "range": [
                                        [1, 2],
                                        [3, 3],
                                        [1024, 4096],
                                        [1024, 4096]
                                    ],
                                    "shape": [-1, 3, -1, -1]
                                }]
                            }]
                        }]
        get_context().add_addition("missing_support_info", json.dumps(missing_info))
        input_list = [
            {
                'shape': (1, 1, 1080, 1080, 16),
                'ori_shape': (1, 3, 1080, 1080),
                'ori_format': 'NCHW',
                'format': 'NC1HWC0',
                'dtype': 'float16',
                'range': ((1, 2), (1, 1), (1024, 4096), (1024, 4096), (16, 16)),
                'ori_range': ((1, 2), (3, 3), (1024, 4096), (1024, 4096))
            }, {
                'ori_shape': (64, 3, 7, 7),
                'ori_format': 'NCHW',
                'format': 'FRACTAL_Z',
                'dtype': 'float16'
            }, None, None, {
                'shape': (-1, 4, -1, -1, 16),
                'ori_shape': (-1, 64, -1, -1),
                'ori_format': 'NCHW',
                'format': 'NC1HWC0',
                'dtype': 'float16'
            }, (1, 1, 2, 2), (3, 3, 3, 3), (1, 1, 1, 1), 1, 'NCHW', 0, 'test_conv2d_fuzz_build_correct_range']
        ret = conv2d_generalization(*input_list)
        conv2d(*(ret[0]))
print("adding conv2d test_conv2d_fuzz_build_correct_range testcase")
# ut_case.add_cust_test_func(test_func=test_conv2d_fuzz_build_correct_range)

def test_conv2d_split_info_unknown_shape(test_arg):
    input_list = [
        {
            'ori_shape': (-2,),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW']
    get_op_support_info(*input_list)
print("adding conv2d test_conv2d_split_info_unknown_shape testcase")
ut_case.add_cust_test_func(test_func=test_conv2d_split_info_unknown_shape)

def test_conv2d_split_info_split_n(test_arg):
    input_list = [
        {
            'ori_shape': (-1, 64, 32, 32),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW']
    get_op_support_info(*input_list)
print("adding conv2d test_conv2d_split_info_split_n testcase")
ut_case.add_cust_test_func(test_func=test_conv2d_split_info_split_n)

def test_conv2d_split_info_split_h(test_arg):
    input_list = [
        {
            'ori_shape': (1, 64, -1, 32),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW']
    get_op_support_info(*input_list)
print("adding conv2d test_conv2d_split_info_split_h testcase")
ut_case.add_cust_test_func(test_func=test_conv2d_split_info_split_h)

def test_conv2d_split_info_split_w(test_arg):
    input_list = [
        {
            'ori_shape': (1, 64, 32, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW']
    get_op_support_info(*input_list)
print("adding conv2d test_conv2d_split_info_split_w testcase")
ut_case.add_cust_test_func(test_func=test_conv2d_split_info_split_w)

def test_conv2d_split_info_split_nhw(test_arg):
    input_list = [
        {
            'ori_shape': (-1, 64, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW']
    get_op_support_info(*input_list)
print("adding conv2d test_conv2d_split_info_split_nhw testcase")
ut_case.add_cust_test_func(test_func=test_conv2d_split_info_split_nhw)

def test_conv2d_split_info_split_bias(test_arg):
    input_list = [
        {
            'ori_shape': (-1, 64, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'format': 'ND',
            'dtype': 'float16'
        }, None, {
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW']
    get_op_support_info(*input_list)
print("adding conv2d test_conv2d_split_info_split_bias testcase")
ut_case.add_cust_test_func(test_func=test_conv2d_split_info_split_bias)

def test_invalid_data_rm_1(test_arg):
    inputs = tvm.placeholder((1, 1, 8, 8, 32), name="fmap", dtype="int8",
                             attrs={"ori_shape": (1, 32, 8, 8), "format": "NCHW", "ori_fomat": "NCHW"})
    weights = tvm.placeholder((4, 2, 16, 32), name="weight", dtype="int8",
                              attrs={"ori_shape": (32, 32, 2, 2), "format": "FRACTAL_Z", "ori_format": "NCHW"})
    para_dict = {'pad_h': [0, 0], 'pad_w': [0, 0], 'stride_h': 2, 'stride_w': 2, 'dilate_h': 1, 'dilate_w': 1,
                 'offset_x': 0, 'filter_h': 3, 'filter_w': 3, 'bias_tensor': None, 'offset_w_tensor': None,
                 'fusion_para': {'fmap_l1_addr_flag': -1, 'fmap_l1_valid_size': -1, 'slice_offset': []},
                 'kernel_name': 'te_fused_op_conv2d_trans_data_27b023e523088153_7d02bbc69c3cd8f9_0', 'group': 1,
                 'enlarge': 1, 'c1_opt': 1, 'cout1_opt': 1, 'group_opt': 1, 'a_shape': [100, 1, 15, 33, 16],
                 'weight_fracz_shape': [9, 1, 16, 16], 'weight_ori_shape_nchw': [1, 16, 3, 3],
                 'multi_conv2d_fusion_flag': False}
    optim_dict = {'c0_optim_flg': False, 'use_v200_c04_flg': False, 'v220_c04_mode': 'disabled',
                  'invalid_data_rm': True}
    res = conv(inputs, weights, para_dict, optim_dict, dsl_flag=False)
print("adding conv2d test_invalid_data_rm_1 testcase")
ut_case.add_cust_test_func(test_func=test_invalid_data_rm_1)  

def test_invalid_data_rm_2(test_arg):
    inputs = tvm.placeholder((1, 1, 8, 8, 32), name="fmap", dtype="int8",
                             attrs={"ori_shape": (1, 32, 8, 8), "format": "NCHW", "ori_fomat": "NCHW"})
    weights = tvm.placeholder((4, 2, 16, 32), name="weight", dtype="int8",
                              attrs={"ori_shape": (32, 32, 2, 2), "format": "FRACTAL_Z", "ori_format": "NCHW"})
    para_dict = {'pad_h': [0, 0], 'pad_w': [0, 0], 'stride_h': 2, 'stride_w': 2, 'dilate_h': 1, 'dilate_w': 1,
                 'offset_x': 0, 'filter_h': 3, 'filter_w': 3, 'bias_tensor': None, 'offset_w_tensor': None,
                 'fusion_para': {'fmap_l1_addr_flag': -1, 'fmap_l1_valid_size': -1, 'slice_offset': []},
                 'kernel_name': 'te_fused_op_conv2d_trans_data_27b023e523088153_7d02bbc69c3cd8f9_0', 'group': 1,
                 'enlarge': 1, 'c1_opt': 1, 'cout1_opt': 1, 'group_opt': 1, 'a_shape': [100, 1, 15, 33, 16],
                 'weight_fracz_shape': [9, 1, 16, 16], 'weight_ori_shape_nchw': [1, 16, 3, 3],
                 'multi_conv2d_fusion_flag': False}
    optim_dict = {'c0_optim_flg': False, 'use_v200_c04_flg': False, 'v220_c04_mode': 'disabled',
                  'invalid_data_rm': False}
    res = conv(inputs, weights, para_dict, optim_dict, dsl_flag=False)
print("adding conv2d test_invalid_data_rm_2 testcase")
ut_case.add_cust_test_func(test_func=test_invalid_data_rm_2)  

if __name__ == '__main__':
    ut_case.run(["Ascend910", "Ascend710", "Ascend310"])
    dsl_cpu_test_int8()
    dsl_cpu_test_fp16()
    exit(0)
