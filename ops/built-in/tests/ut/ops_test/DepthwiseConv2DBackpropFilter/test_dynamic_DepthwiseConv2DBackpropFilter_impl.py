#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("DepthwiseConv2DBackpropFilter", "impl.dynamic.depthwise_conv2d_backprop_filter",
               "depthwise_conv2d_backprop_filter")
dynamic_conv2d_bp_filter_op_testcase = [
    # success
    ((3, 40, 1, 2), (3, 40, 1, 2), (40, 1, 3, 3), [1,1,1,1], [1,1,1,1], (1, 1, 1, 1), "NCHW", [0], "success", "dynamic_conv2d_bp_filter_op_testcase_0"),
    ((3, 200, 75, 7), (3, 198, 37, 7), (3, 3, 1, 7), [1,1,2,1], [1,1,1,1], (-1, -1, -1, -1), "NHWC", [2, 3], "success", "dynamic_conv2d_bp_filter_op_testcase_1"),
    ((3, 200, 75, 40), (3, 198, 37, 40), (3, 3, 1, 40), [1,1,2,1], [1,1,1,1], (0, 0, 0, 0), "NHWC", [0, 2, 3], "success", "dynamic_conv2d_bp_filter_op_testcase_2"),
    # h -1
    ((3, 4, 2, 40), (3, 1, 1, 40), (2, 2, 1, 40), [1,2,1,1], [1,1,1,1], (0, 0, 0, 0), "NHWC", [2], "success", "dynamic_conv2d_bp_filter_op_testcase_15"),
    ((-1, 2, 2, -1), (-1, 2, 2, -1), (2, 1, 3, 3), [1,1,1,2], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0,1,3], "success", "dynamic_conv2d_bp_filter_op_testcase_19"),
    # -2
    ((3, 40, 200, 75), (3, 40, 200, 75), (40, 1, 9, 9), [1,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [-2], "success", "dynamic_conv2d_bp_filter_op_testcase_14"),
    # dynamic_nh in input, dynamic_hw in dedy
    ((3, 40, 200, 75), (3, 40, 200, 75), (40, 1, 9, 9), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [[0,2],[2,3]], "success", "dynamic_conv2d_bp_filter_op_testcase_16"),
    # dynamic_w in input, dynamic_hw in dedy
    ((3, 40, 200, 75), (3, 40, 200, 75), (40, 1, 9, 9), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [[3],[2,3]], "success", "dynamic_conv2d_bp_filter_op_testcase_17"),
    ((3, 40, 200, 75), (3, 40, 200, 75), (40, 1, 9, 9), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [[2,3],[3]], "success", "dynamic_conv2d_bp_filter_op_testcase_18"),
    # dedy_c != fmap_c * filter_n
    ((3, 40, 200, 75), (3, 41, 200, 75), (40, 1, 9, 9), [1,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_3"),
    # filter_c != fmap_c
    ((3, 40, 200, 75), (3, 40, 200, 75), (13, 1, 9, 9), [1,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_4"),
    # fmap_n != dedy_n
    ((3, 40, 200, 75), (4, 40, 200, 75), (40, 1, 9, 9), [1,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_5"),
    # stride_n != 1
    ((3, 40, 200, 75), (3, 40, 200, 75), (40, 1, 9, 9), [2,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_6"),
    # strides dim != 4
    ((3, 40, 200, 75), (3, 40, 200, 75), (40, 1, 9, 9), [1,1,1,1,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_7"),
    # pads dim != 4
    ((3, 40, 200, 75), (3, 40, 200, 75), (40, 1, 9, 9), [1,1,1,1], [1,1,1,1], (0, 0, 0), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_8"),
    # stride_h/w > 63
    ((3, 40, 200, 75), (3, 40, 200, 75), (40, 1, 9, 9), [1,64,64,1], [1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_9"),
    # dilations != [1,1,1,1]
    ((3, 40, 200, 75), (3, 40, 200, 75), (40, 1, 9, 9), [1,1,1,1], [1,2,2,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_10"),
    # dilations dim != 4
    ((3, 40, 200, 75), (3, 40, 200, 75), (40, 1, 9, 9), [1,1,1,1], [1,1,1,1,1], (-1, -1, -1, -1), "NCHW", [0, 2, 3], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_11"),
    # dedy_h does not match fmap_h
    ((3, 40, 200, 75), (3, 40, 200, 75), (40, 1, 9, 9), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [0], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_12"),
    # dedy_h less than 2
    ((3, 40, 9, 75), (3, 40, 200, 75), (40, 1, 9, 9), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [0], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_13"),
    # filter_n not equal 2
    ((3, 40, 9, 75), (3, 40, 200, 75), (40, 40, 9, 9), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [0], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_14"),
    # filter_h equal -1
    ((3, 40, 9, 75), (3, 40, 200, 75), (40, 1, -1, 9), [1,1,1,1], [1,1,1,1], (0, 0, 0, 0), "NCHW", [0], RuntimeError, "dynamic_conv2d_bp_filter_op_testcase_15"),
    # min_range_negative
    ((4, 2, 16, 1), (4, 2, 23, 1), (1, 8, 1, 1), [1, 1, 1, 1], [1, 1, 1, 1], (0, 0, 0, 0), "NHWC", [-2], "success", "dynamic_conv2d_bp_filter_op_testcase_min_range_negative")
]

def _get_kernel_name(x_shape, dedy_shape, filter_shape, strides, dilations, pads, data_format):
    padding = "SAME" if -1 in pads else "VALID"
    kernel_name = 'dp_conv2d_bp_filter' + '_'.join(map(str, x_shape)) + '_' + '_'.join(map(str, dedy_shape)) + '_' + '_'.join(
        map(str, filter_shape)) + '_' + str(strides[0]) + '_' + str(dilations[0]) + "_" + padding + '_' + data_format
    return kernel_name


def _shape_to_NC1HWC0(shape, data_format, dtype):
    if data_format.upper() == "NCHW":
        n, c, h, w = shape
    else:  # NCHW
        n, h, w, c = shape
    c0 = 16 if dtype.lower() == "float16" else 32
    c1 = (c + c0 - 1) // c0
    return (n, c1, h, w, c0)


def _shape_to_C1HWNCoC0(shape, data_format, dtype):
    if data_format.upper() == "HWCN":
        h, w, c, n = shape
    else:  # NCHW
        n, c, h, w = shape
    c0 = 16 if dtype.lower() == "float16" else 32
    c1 = (c + c0 - 1) // c0
    return (c1, h, w, n, c0, c0)


def _get_range_from_shape(shape, dynamic_dim):
    ori_range = [(dim, dim) for dim in shape]
    if dynamic_dim:
        for dim in dynamic_dim:
            if shape[dim] == -1:
                ori_range[dim] = (max(1, shape[dim] // 2), None)
            else:
                ori_range[dim] = (max(1, shape[dim] // 2), min(4096, shape[dim] * 2))
    return ori_range


def _trans_dynamic_shape(shape, data_format, dynamic_dim):
    shape = list(shape)
    if 0 in dynamic_dim:
        n_dim = data_format.index("N")
        shape[n_dim] = -1
    if 1 in dynamic_dim:
        if data_format == 'NC1HWC0':
            shape[1] = -1
        else:
            c_dim = data_format.index("C")
            shape[c_dim] = -1
    if 2 in dynamic_dim:
        h_dim = data_format.index("H")
        shape[h_dim] = -1
    if 3 in dynamic_dim:
        w_dim = data_format.index("W")
        shape[w_dim] = -1
    return tuple(shape)


def _gen_case(param):
    input_ori_shape, out_backprop_ori_shape, filter_size, strides, dilations, pads, data_format, dynamic_dim, expect_result, kernel_name = param

    data_format = data_format.upper()
    dtype = "float16"

    x_unknown_rank_flag = False
    dy_unknown_rank_flag = False
    if isinstance(dynamic_dim[0], list):
        if dynamic_dim[0][0] == -2:
            x_unknown_rank_flag = True
        elif dynamic_dim[1][0] == -2:
            dy_unknown_rank_flag = True
    elif dynamic_dim[0] == -2:
        x_unknown_rank_flag = True
        dy_unknown_rank_flag = True

    input_shape = _shape_to_NC1HWC0(input_ori_shape, data_format, dtype)
    out_backprop_shape = _shape_to_NC1HWC0(out_backprop_ori_shape, data_format, dtype)
    filter_format = 'NCHW' if data_format == 'NCHW' else 'HWCN'
    filter_grad_shape = _shape_to_C1HWNCoC0(filter_size, filter_format, dtype)
    
    x = {
        "shape": _trans_dynamic_shape(input_shape, "NCHWI", dynamic_dim[0] if isinstance(dynamic_dim[0], list) else dynamic_dim),
        "format": "NC1HWC0",
        "ori_shape": [-2] if x_unknown_rank_flag else _trans_dynamic_shape(input_ori_shape, data_format,
                     dynamic_dim[0] if isinstance(dynamic_dim[0], list) else dynamic_dim),
        "ori_format": data_format,
        "dtype": dtype,
        "range": _get_range_from_shape(input_shape, dynamic_dim[0] if isinstance(dynamic_dim[0], list) else dynamic_dim)
    }
    filter_shape = {
        "shape": [1,1,1,1],
        "ori_shape": [1,1,1,1],
        "ori_format": filter_format,
        "format": "C1HWNCoC0",
        "dtype": "int32",
        "range": _get_range_from_shape(filter_size, [])
    }
    out_backprop = {
        "shape": _trans_dynamic_shape(out_backprop_shape, "NCHWI", dynamic_dim[1] if isinstance(dynamic_dim[0], list) else dynamic_dim),
        "format": "NC1HWC0",
        "ori_shape": [-2] if dy_unknown_rank_flag else _trans_dynamic_shape(out_backprop_ori_shape, data_format,
                     dynamic_dim[1] if isinstance(dynamic_dim[0], list) else dynamic_dim),
        "ori_format": data_format,
        "dtype": dtype,
        "range": _get_range_from_shape(out_backprop_shape, dynamic_dim[1] if isinstance(dynamic_dim[0], list) else dynamic_dim)
    }
    filter_grad = {
        "shape": filter_grad_shape,
        "format": "C1HWNCoC0",
        "ori_shape": filter_size,
        "ori_format": filter_format,
        "dtype": "float32",
        "range": _get_range_from_shape(filter_grad_shape, [])
    }

    kernel_name = kernel_name
    return {
        "params": [x, filter_shape, out_backprop, filter_grad, strides, dilations, pads, data_format],
        "case_name": kernel_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }


for case in dynamic_conv2d_bp_filter_op_testcase:
    ut_case.add_case(["Ascend910A"], _gen_case(case))


def test_depthwise_conv2d_backprop_filter_fuzz_build_generalization_general(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_filter import depthwise_conv2d_backprop_filter_generalization
    input_list = [
        {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (3, 1, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
        'depthwise_conv2d_backprop_filter_fuzz_build_generalization_general', {"mode": "keep_rank"}]
    depthwise_conv2d_backprop_filter_generalization(*input_list)

ut_case.add_cust_test_func('Ascend910A', test_func=test_depthwise_conv2d_backprop_filter_fuzz_build_generalization_general)


def test_depthwise_conv2d_backprop_filter_fuzz_build_generalization_range_max_fixed(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_filter import depthwise_conv2d_backprop_filter_generalization
    input_list = [
        {
            'shape': (50, 1, 35, 2896, 16),
            'ori_shape': (50, 2, 35, 2896),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'shape': (50, 1, 26, 2888, 16),
            'ori_shape': (50, 2, 26, 2888),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (2, 1, 10, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        },  (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
        'depthwise_conv2d_backprop_filter_fuzz_build_generalization_range_max_fixed']
    depthwise_conv2d_backprop_filter_generalization(*input_list)

ut_case.add_cust_test_func('Ascend910A', test_func=test_depthwise_conv2d_backprop_filter_fuzz_build_generalization_range_max_fixed)

def test_depthwise_conv2d_backprop_filter_fuzz_build_w_range_max_fixed(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_filter import depthwise_conv2d_backprop_filter_generalization
    input_list = [
        {
            'shape': (1, 1, 8, 3051, 16),
            'ori_shape': (1, 8, 3051, 1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'shape': (1, 1, 1, 3040, 16),
            'ori_shape': (1, 1, 3040, 1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (8, 12, 1, 1),
            'ori_format': 'HWCN',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        },  (1, 1, 3, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
        'test_depthwise_conv2d_backprop_filter_fuzz_build_w_range_max_fixed']
    depthwise_conv2d_backprop_filter_generalization(*input_list)

ut_case.add_cust_test_func('Ascend910A', test_func=test_depthwise_conv2d_backprop_filter_fuzz_build_w_range_max_fixed)

def test_depthwise_conv2d_backprop_filter_fuzz_build_dedy_h_equal_one_w_range_max_fixed(test_arg):
    from impl.dynamic.depthwise_conv2d_backprop_filter import depthwise_conv2d_backprop_filter_generalization
    input_list = [
        {
            'shape': (6, 1, 2, 2857, 16),
            'ori_shape': (6, 2, 2857, 1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'shape': (6, 1, 1, 2857, 16),
            'ori_shape': (6, 1, 2857, 1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (11, 1, 1, 1),
            'ori_format': 'HWCN',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        },  (1, 1, 2, 1), (1, 1, 1, 1), (4, 5, 0, 0), 'NCHW',
        'test_depthwise_conv2d_backprop_filter_fuzz_build_w_range_max_fixed']
    depthwise_conv2d_backprop_filter_generalization(*input_list)

ut_case.add_cust_test_func('Ascend910A', test_func=test_depthwise_conv2d_backprop_filter_fuzz_build_dedy_h_equal_one_w_range_max_fixed)

if __name__ == '__main__':
    ut_case.run()
    exit(0)
