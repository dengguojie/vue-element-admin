#!/usr/bin/env python
# -*- coding:utf-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AvgPoolV2", "impl.dynamic.avg_pool_v2", "avg_pool_v2")

avgpoolv2_ut_testcases = [
    # fmap_shape, filter_shape, ksize, strides, padding, pads, data_format, ceil_mode, exclusive, range, expect_result
    # custom cases H/W dimensions dynamic
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "CALCULATED", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "CALCULATED", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),

    ((1, 32, -1, -1), (32, 1, 7, 7), (1, 1, 7, 7), (1, 1, 2, 2), "CALCULATED", (1, 1, 2, 2), "NCHW", True, False, 
    [(1,1), (32,32), (1,500), (1,50)], "success"),
    ((1, 32, -1, -1), (32, 1, 8, 8), (1, 1, 8, 8), (1, 1, 1, 1), "CALCULATED", (4, 5, 6, 7), "NCHW", True, False, 
    [(1,1), (32,32), (1,70), (50,200)], "success"),
    ((1, 32, -1, -1), (32, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,85)], "success"),
    # N/H/W dimensions dynamic
    ((-1, 32, -1, 56), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,10), (32,32), (1,100), (56,56)], "success"),
    ((-1, 32, 100, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,10), (32,32), (100,100), (1,200)], "success"),
    ((-1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,10), (32,32), (1,100), (1,100)], "success"),
    # batch dynamic, h/w static
    ((-1, 32, 100, 56), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,10), (32,32), (100,100), (56,56)], "success"),
    ((-1, 128, 768, 680), (128, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,10), (128,128), (768,768), (680,680)], "success"),
    ((-1, 512, 1024, 1024), (512, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "CALCULATED", (2, 2, 2, 2), "NCHW", True, False, 
    [(1,10), (512,512), (1024,1024), (1024,1024)], "success"),

    # load2d
    ((-1, 512, 1024, 1024), (32, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), "CALCULATED", (0,0,0,0), "NCHW", False, False, 
    [(1,10), (512,512), (1024,1024), (1024,1024)], "success"),

    # range [1, None]
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,None), (1,None)], "success"),

    # static shape, need at least one dimension in N/H/W is a variable
     ((1, 32,56, 56), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (56,56), (56,56)], RuntimeError),

    # channel = -1
    ((1, -1, 56, 56), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (56,56), (56,56)], RuntimeError),
    # invalid ksize/strides/pads
    ((1, 32, -1, -1), (32, 1, 3, 3), (3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    ((1, 32, -1, -1), (32, 1, 3, 3), (2, 2, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (3, 3, 3, 3), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, -1, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    # invalid padding_mode
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "EXCUTE", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    # invalid format
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "HWCN", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),

    # invalid strides
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 64, 64), "SAME", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    # invalid ksize
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 256, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    # invalid pads
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 301, 3), (1, 1, 2, 2), "CALCULATED", (300, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),

    # kuaishou network
    # blocky
    ((16, 3, -1, -1), (3, 1, 3, 3), (1, 1, 3, 3), (1, 1, 1, 1), "CALCULATED", (1, 1, 1, 1), "NCHW", False, False, 
    [(16,16), (3,3), (96,896), (72,672)], "success"),
    # blur
    ((16, 3, -1, -1), (3, 1, 3, 3), (1, 1, 3, 3), (1, 1, 1, 1), "CALCULATED", (1, 1, 1, 1), "NCHW", False, False, 
    [(16,16), (3,3), (512,1200), (512,1200)], "success"),
    # defocus
    ((16, 3, -1, -1), (3, 1, 3, 3), (1, 1, 3, 3), (1, 1, 1, 1), "CALCULATED", (1, 1, 1, 1), "NCHW", False, False, 
    [(16,16), (3,3), (256,512), (256,512)], "success"),
    # noise
    ((16, 3, -1, -1), (3, 1, 3, 3), (1, 1, 3, 3), (1, 1, 1, 1), "CALCULATED", (1, 1, 1, 1), "NCHW", False, False, 
    [(16,16), (3,3), (96,448), (72,448)], "success"),


    # vector
    ((1, 32, -1, -1), None, (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),

]

def get_kernel_name(fmap_shape, ksize, strides, padding, pads, data_format, ceil_mode, exclusive, in_range):

    batch_range, c_range, h_range, w_range = in_range
    fmap_info = "_".join([str(i) for i in fmap_shape])
    ksize_info = "_".join([str(i) for i in ksize])
    strides_info = "_".join([str(i) for i in strides])
    pads_info = "_".join([str(i) for i in pads])
    range_info = "_".join([str(batch_range[0]), str(batch_range[1]), str(c_range[0]), str(c_range[1]),
                        str(h_range[0]), str(h_range[1]), str(w_range[0]), str(w_range[1])])

    kernel_name = "fmap_{}_k_{}_s_{}_padding_{}_pads_{}_format_{}_ceilmode_{}_exclusive_{}_range_{}".format(fmap_info,
                    ksize_info, strides_info, padding, pads_info, data_format, str(ceil_mode), str(exclusive), range_info)

    return kernel_name


def gen_avgpoolv2_params(case):
    fmap_shape, filter_shape, ksize, strides, padding, pads, data_format, ceil_mode, exclusive, in_range, expect_result = case

    kernel_name = get_kernel_name(fmap_shape, ksize, strides, padding, pads, data_format, ceil_mode, exclusive, in_range)

    output_shape = fmap_shape
    if filter_shape:
        filter_range = [(filter_shape[0], filter_shape[0]), (filter_shape[1], filter_shape[1]), \
                        (filter_shape[2], filter_shape[2]), (filter_shape[3], filter_shape[3])]
        weight = {"ori_shape": filter_shape, "dtype": "float16", "ori_format": data_format, "range": filter_range}
    else:
        weight = None

    inputs = {"ori_shape": fmap_shape, "dtype": "float16", "ori_format": data_format, "range": in_range}
    outputs = {"ori_shape": output_shape, "dtype": "float16", "ori_format": data_format}

    global_pooling = False
    op_params = [inputs, weight, outputs, ksize, strides, padding, pads, data_format, global_pooling, \
                ceil_mode, exclusive]

    return {"params": op_params,
            "expect": expect_result,
            "support_expect": True}

# for testcase in avgpoolv2_ut_testcases:
#     ut_case.add_case(["Ascend910A"], gen_avgpoolv2_params(testcase))
#     ut_case.add_case(["Ascend710"], gen_avgpoolv2_params(testcase))
#     ut_case.add_case(["Ascend310"], gen_avgpoolv2_params(testcase))

from impl.dynamic.avg_pool_v2 import avg_pool_v2_generalization

# output_h lower than 1
# format NCHW, valid mode, kh is 17, larger than ori_range h lowest 16
def test_avg_pool_v2_graph_mode_output_h_format_NCHW(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (16, 31), (64, 127)),
            'ori_range': ((1, 1), (16, 31), (16, 31), (64, 127))
        }, {
            # weight
            'ori_shape': (16, 1, 17, 17),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            "range": [(16, 16), (1, 1), (17, 17), (17, 17)]
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 1, 17, 17), (1, 1, 1, 1), "VALID", None, 'NCHW', None, None, None, 0,
        'avg_pool_v2_graph_mode_output_h_format_NCHW']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["lower_limit"]}}]:
        raise Exception("test_avg_pool_v2_graph_mode_output_h_format_NCHW failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_output_h_format_NCHW testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_output_h_format_NCHW)

# output_w lower than 1
# format NCHW, valid mode, kw is 17, larger than ori_range w lowest 16
def test_avg_pool_v2_graph_mode_output_w_format_NCHW(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (16, 31)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 17, 17),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            "range": [(16, 16), (1, 1), (17, 17), (17, 17)]
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 1, 17, 17), (1, 1, 1, 1), "VALID", None, 'NCHW', None, None, None, 0,
        'avg_pool_v2_graph_mode_output_w_format_NCHW']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["lower_limit"]}}]:
        raise Exception("test_avg_pool_v2_graph_mode_output_w_format_NCHW failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_output_w_format_NCHW testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_output_w_format_NCHW)

# output_h lower than 1
# format NHWC, valid mode, kh is 17, larger than ori_range h lowest 16
def test_avg_pool_v2_graph_mode_output_h_format_NHWC(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (16, 31)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 17, 17),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            "range": [(16, 16), (1, 1), (17, 17), (17, 17)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 17, 17, 1), (1, 1, 1, 1), "VALID", None, 'NHWC', None, None, None, 0,
        'avg_pool_v2_graph_mode_output_h_format_NHWC']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["lower_limit"]}}]:
        raise Exception("test_avg_pool_v2_graph_mode_output_h_format_NHWC failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_output_h_format_NHWC testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_output_h_format_NHWC)

# output_w lower than 1
# format NHWC, valid mode, kw is 17, larger than ori_range w lowest 16
def test_avg_pool_v2_graph_mode_output_w_format_NHWC(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 17, 17),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            "range": [(16, 16), (1, 1), (17, 17), (17, 17)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 17, 17, 1), (1, 1, 1, 1), "VALID", None, 'NHWC', None, None, None, 0,
        'avg_pool_v2_graph_mode_output_w_format_NHWC']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["lower_limit"]}}]:
        raise Exception("test_avg_pool_v2_graph_mode_output_w_format_NHWC failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_output_w_format_NHWC testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_output_w_format_NHWC)

# limit_size larger than l1_size
# format NCHW, valid mode, kw is 1, kh is 16, ori_range w largest is 2047
def test_avg_pool_v2_graph_mode_l1_size_NCHW_valid(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (1024, 2047)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (1024, 2047))
        }, {
            # weight
            'ori_shape': (16, 1, 16, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (16, 16), (1, 1)]
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 1, 16, 1), (1, 1, 1, 1), "VALID", None, 'NCHW', None, None, None, 0,
        'avg_pool_v2_graph_mode_l1_size_NCHW_valid']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["upper_limit"]}}]:
        raise Exception("test_avg_pool_v2_graph_mode_l1_size_NCHW_valid failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_l1_size_NCHW_valid testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_l1_size_NCHW_valid)

# limit_size larger than l1_size
# format NHWC, valid mode, kw is 1, kh is 16, ori_range w largest is 2047
def test_avg_pool_v2_graph_mode_l1_size_NHWC_valid(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (1024, 2047), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (1024, 2047), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 16, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (16, 16), (1, 1)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 16, 1, 1), (1, 1, 1, 1), "VALID", None, 'NHWC', None, None, None, 0,
        'avg_pool_v2_graph_mode_l1_size_NHWC_valid']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["upper_limit"]}}]:
        raise Exception("test_avg_pool_v2_graph_mode_l1_size_NHWC_valid failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_l1_size_NHWC_valid testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_l1_size_NHWC_valid)

# limit_size larger than l1_size
# format NCHW, same mode, kw is 1, kh is 16, ori_range w largest is 2047
def test_avg_pool_v2_graph_mode_l1_size_NCHW_same(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (1024, 2047)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (1024, 2047))
        }, {
            # weight
            'ori_shape': (16, 1, 16, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (16, 16), (1, 1)]
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 1, 16, 1), (1, 1, 1, 1), "SAME", None, 'NCHW', None, None, None, 0,
        'avg_pool_v2_graph_mode_l1_size_NCHW_same']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["upper_limit"]}}]:
        raise Exception("test_avg_pool_v2_graph_mode_l1_size_NCHW_same failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_l1_size_NCHW_same testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_l1_size_NCHW_same)

# limit_size larger than l1_size
# format NHWC, same mode, kw is 1, kh is 16, ori_range w largest is 2047
def test_avg_pool_v2_graph_mode_l1_size_NHWC_same(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (1024, 2047), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (1024, 2047), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 16, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (16, 16), (1, 1)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 16, 1, 1), (1, 1, 1, 1), "SAME", None, 'NHWC', None, None, None, 0,
        'avg_pool_v2_graph_mode_l1_size_NHWC_same']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["upper_limit"]}}]:
        raise Exception("test_avg_pool_v2_graph_mode_l1_size_NHWC_same failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_l1_size_NHWC_same testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_l1_size_NHWC_same)

# limit_size larger than l1_size
# format NCHW, same mode, kw is 1, kh is 16, ori_range w largest is 2047
def test_avg_pool_v2_graph_mode_l1_size_NCHW_calculated(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (1024, 2047)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (1024, 2047))
        }, {
            # weight
            'ori_shape': (16, 1, 16, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (16, 16), (1, 1)]
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 1, 16, 5), (1, 1, 1, 1), "CALCULATED", (1, 1, 2, 2), 'NCHW', None, None, None, 0,
        'avg_pool_v2_graph_mode_l1_size_NCHW_calculated']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["upper_limit"]}}]:
        raise Exception("test_avg_pool_v2_graph_mode_l1_size_NCHW_calculated failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_l1_size_NCHW_calculated testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_l1_size_NCHW_calculated)

# limit_size larger than l1_size
# format NHWC, calculated mode, kw is 1, kh is 16, ori_range w largest is 2047
def test_avg_pool_v2_graph_mode_l1_size_NHWC_calculated(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (1024, 2047), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (1024, 2047), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 16, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (16, 16), (1, 1)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x
        (1, 16, 5, 1), (1, 1, 1, 1), "CALCULATED", (1, 1, 2, 2), 'NHWC', None, None, None, 0,
        'avg_pool_v2_graph_mode_l1_size_NHWC_calculated']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["upper_limit"]}}]:
        raise Exception("test_avg_pool_v2_graph_mode_l1_size_NHWC_calculated failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_l1_size_NHWC_calculated testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_l1_size_NHWC_calculated)

# generalize_config unsupported
# raise error
def test_avg_pool_v2_generalize_config_unsupported(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 1, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (1, 1), (1, 1)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x, generalize_config
        (1, 1, 1, 1), (1, 1, 1, 1), "VALID", None, 'NHWC', None, None, None, 0,
        'avg_pool_v2_generalize_config_unsupported', {"mode": "keep"}]
    try:
        avg_pool_v2_generalization(*input_list)
    except RuntimeError:
        print("expected")
        pass
print("adding avg_pool_v2 test_avg_pool_v2_generalize_config_unsupported testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_generalize_config_unsupported)

# ori_shape is unknown rank
# raise error
def test_avg_pool_v2_unknown_rank(test_arg):
    input_list = [
        {
            # inputs
            'shape': [-2],
            'ori_shape': [-2],
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 1, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (1, 1), (1, 1)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x, generalize_config
        (1, 1, 1, 1), (1, 1, 1, 1), "VALID", None, 'NHWC', None, None, None, 0,
        'avg_pool_v2_unknown_rank']
    try:
        avg_pool_v2_generalization(*input_list)
    except RuntimeError:
        print("expected")
        pass
print("adding avg_pool_v2 test_avg_pool_v2_unknown_rank testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_unknown_rank)

# unsupported format
# raise error
def test_avg_pool_v2_unsupported_format(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'HWCN',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 1, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (1, 1), (1, 1)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x, generalize_config
        (1, 1, 1, 1), (1, 1, 1, 1), "VALID", None, 'NHWC', None, None, None, 0,
        'avg_pool_unsupported_format']
    try:
        avg_pool_v2_generalization(*input_list)
    except RuntimeError:
        print("expected")
        pass
print("adding avg_pool_v2 test_avg_pool_v2_unsupported_format testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_unsupported_format)

# supported range
# NCHW, valid
def test_avg_pool_v2_graph_mode_supported_range_NCHW_valid(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (16, 31)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x, generalize_config
        (1, 1, 3, 3), (1, 1, 1, 1), "VALID", None, 'NCHW', None, None, None, 0,
        'avg_pool_v2_graph_mode_supported_range_NCHW_valid']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_v2_graph_mode_supported_range_NCHW_valid failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_supported_range_NCHW_valid testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_supported_range_NCHW_valid)

# supported range
# NCHW, same
def test_avg_pool_v2_graph_mode_supported_range_NCHW_same(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (16, 31)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x, generalize_config
        (1, 1, 3, 3), (1, 1, 1, 1), "SAME", None, 'NCHW', None, None, None, 0,
        'avg_pool_v2_graph_mode_supported_range_NCHW_same']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_v2_graph_mode_supported_range_NCHW_same failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_supported_range_NCHW_same testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_supported_range_NCHW_same)

# supported range
# NCHW, calculated
def test_avg_pool_v2_graph_mode_supported_range_NCHW_calculated(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (16, 31)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x, generalize_config
        (1, 1, 3, 3), (1, 1, 1, 1), "CALCULATED", (1, 1, 2, 2), 'NCHW', None, None, None, 0,
        'avg_pool_v2_graph_mode_supported_range_NCHW_calculated']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_v2_graph_mode_supported_range_NCHW_calculated failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_supported_range_NCHW_calculated testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_supported_range_NCHW_calculated)

# supported range
# NHWC, valid
def test_avg_pool_v2_graph_mode_supported_range_NHWC_valid(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x, generalize_config
        (1, 3, 3, 1), (1, 1, 1, 1), "VALID", None, 'NHWC', None, None, None, 0,
        'avg_pool_v2_graph_mode_supported_range_NHWC_valid']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_v2_graph_mode_supported_range_NCHW_same failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_supported_range_NHWC_valid testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_supported_range_NHWC_valid)

# supported range
# NHWC, same
def test_avg_pool_v2_graph_mode_supported_range_NHWC_same(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x, generalize_config
        (1, 3, 3, 1), (1, 1, 1, 1), "SAME", None, 'NHWC', None, None, None, 0,
        'avg_pool_v2_graph_mode_supported_range_NHWC_same']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_v2_graph_mode_supported_range_NHWC_same failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_supported_range_NHWC_same testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_supported_range_NHWC_same)

# supported range
# NHWC, calculated
def test_avg_pool_v2_graph_mode_supported_range_NHWC_calculated(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x, generalize_config
        (1, 3, 3, 1), (1, 1, 1, 1), "CALCULATED", (1, 1, 2, 2), 'NHWC', None, None, None, 0,
        'avg_pool_v2_graph_mode_supported_range_NHWC_calculated']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_v2_graph_mode_supported_range_NHWC_calculated failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_supported_range_NHWC_calculated testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_supported_range_NHWC_calculated)

# supported range
# range includes "None"
def test_avg_pool_v2_graph_mode_supported_range_none(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, None), (16, None), (16, 31)),
            'ori_range': ((1, 1), (64, None), (16, None), (16, 31))
        }, {
            # weight
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            'range': [(16, 16), (1, 1), (3, 3), (3, 3)]
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, pads, data_format, global_pooling, ceil_mode, exclusive, offset_x, generalize_config
        (1, 3, 3, 1), (1, 1, 1, 1), "SAME", None, 'NHWC', None, None, None, 0,
        'avg_pool_v2_graph_mode_supported_range_none']
    ret = avg_pool_v2_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_v2_graph_mode_supported_range_none failed")
    else:
        print("expected")
print("adding avg_pool_v2 test_avg_pool_v2_graph_mode_supported_range_none testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_graph_mode_supported_range_none)


if __name__ == "__main__":
    ut_case.run("Ascend910A")
    ut_case.run("Ascend710")
    ut_case.run("Ascend310")
    exit(0)