#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.conv2d_backprop_input import get_op_support_info
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2DBackpropInput", "impl.dynamic.conv2d_backprop_input",
               "conv2d_backprop_input")

dynamic_conv2d_bp_input_op_testcase = [
    ((192, 192, 5, 5), (1, 192, -1, 28), (1, 192, -1, 28), (2, 2), (-1, -1, -1, -1), "NCHW", 1, [0, 2], "success"),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 1], "success"),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [1, 3], "success"),
    ((3, 3, 16, 16), (2, 1, 1, 16), (2, 4, 3, 16), (2, 1), (0, 0, 0, 0), "NHWC", 1, [1, 2], "success"),
    ((96, 96, 3, 3), (-1, 96, 2, -1), (-1, 96, 2, -1), (1, 2), (-1, -1, -1, -1), "NCHW", 1, [0, 3], "success"),
    ((3, 3, 256, 256), (2, 34, 32, 256), (2, 36, 34, 256), (1, 1), (0, 0, 0, 0), "NHWC", 1, [0, 2, 3], "success"),
    ((5, 5, 240, 240), (2, 1, 1, 240), (2, 2, 1, 240), (2, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 2, 3], "success"),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 1, 2, 3], "success"),
    ((32, 32, 3, 3), (32, 32, 7, 1), (32, 32, 7, 2), (1, 2), (1, 1, 1, 1), "NCHW", 1, [0], "success"),
    ((1, 32, 1, 1), (1, 1, 4096, 4096), (1, 32, 4096, 4096), (1, 1), (100, 100, 100, 100), "NCHW", 1, [2, 3], "success"),
    ((1, 1, 55, 2), (1, 16, 55, 2), (1, 32, 55, 55), (2, 1), (0, 0, 0, 0), "NHWC", 1, [2, 3], "success"),
    ((16, 7, 3, 3), [-2], (1, 7, 3, 3), (2, 2), (-1, -1, -1, -1), "NCHW", 1, [0, 1, 2, 3], "success"),
    ((7, 6, 64, 10), [-2], (1, 7, 6, 64), (1, 1), (0, 0, 0, 0), "NHWC", 1, [0, 1, 2, 3], "success"),

    ((3, 3, 16, 16), (2, 5, 5, 32), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [0, 2, 3], RuntimeError),
    ((3, 3, 16, 16), (2, 5, 5, 16), (2, 5, 5, 16), (1, 1), (-1, -1, -1, -1), "NHWC", 1, [1], RuntimeError),
]

def _get_kernel_name(filter_shape, dy_shape, x_shape, strides, pads):
    padding = "SAME" if -1 in pads else "VALID"
    if dy_shape == [-2]:
        dy_shape_info = "neg_2"
    else:
        dy_shape_info = '_'.join(map(str, dy_shape))
    kernel_name = 'dynamic_conv2dbp_input_' + '_'.join(map(str, filter_shape)) + '_' + dy_shape_info + '_' + '_'.join(
        map(str, x_shape)) + '_' + '_'.join(map(str, strides))  + "_" + padding
    kernel_name = kernel_name.replace('-1', 'x')
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


def _get_range_from_shape(shape, dynamic_dim=None):
    ori_range = [(dim, dim) for dim in shape]
    if dynamic_dim:
        for dim in dynamic_dim:
            if shape[dim] == -1:
                ori_range[dim] = (max(1, shape[dim] // 2), None)
            else:
                ori_range[dim] = (max(1, shape[dim] // 2), min(4096, shape[dim] * 2))
    return ori_range


def _trans_dynamic_shape(shape, format, dynamic_dim, tran_flag=False):
    shape = list(shape)
    if len(format) == 4:
        if 0 in dynamic_dim:
            n_dim = format.index("N")
            shape[n_dim] = -1
        if 1 in dynamic_dim and tran_flag:
            c_dim = format.index("C")
            shape[c_dim] = -1
        if 2 in dynamic_dim:
            h_dim = format.index("H")
            shape[h_dim] = -1
        if 3 in dynamic_dim:
            w_dim = format.index("W")
            shape[w_dim] = -1
    else:
        if 0 in dynamic_dim:
            shape[0] = -1
        if 1 in dynamic_dim and tran_flag:
            shape[1] = -1
        if 2 in dynamic_dim:
            shape[2] = -1
        if 3 in dynamic_dim:
            shape[3] = -1
    return tuple(shape)


def _gen_trans_data_case(param):
    filter_ori_shape, out_backprop_ori_shape, input_size, strides, pads, data_format, group, dynamic_dim, expect_result = param
    dilations = (1, 1, 1, 1)
    dtype = "float16"
    data_format = data_format.upper()
    filter_format = 'NCHW' if data_format == 'NCHW' else 'HWCN'
    if out_backprop_ori_shape == [-2]:
        input_size_op = [-1, input_size[data_format.index("C")], -1, -1] if data_format == 'NCHW' else [-1, -1, -1, input_size[data_format.index("C")]]
        x = {'shape': [4], 'format': 'NC1HWC0', 'ori_shape': [4], 'ori_format': data_format, 'dtype': 'float16',
             'range': [(1, 4), (1, 2), (2, 10), (2, 10), (16, 16)]}
        filter = {'shape': _shape_to_C1HWNCoC0(filter_ori_shape, filter_format, dtype), 'ori_shape': filter_ori_shape, 'ori_format': filter_format,
                  'format': 'C1HWNCoC0', 'dtype': 'float16', 'range': None}
        out_backprop = {'shape': out_backprop_ori_shape, 'format': 'NC1HWC0', 'ori_shape': out_backprop_ori_shape,
                        'ori_format': data_format, 'dtype': 'float16', 'range': None}
        dx = {'shape': _shape_to_NC1HWC0(input_size_op, data_format, dtype), 'format': 'NC1HWC0',
                      'ori_shape': input_size_op, 'ori_format': data_format, 'dtype': 'float16', 'range': None}
    else:
        filter_shape = _shape_to_C1HWNCoC0(filter_ori_shape, filter_format, dtype),
        out_backprop_shape = _shape_to_NC1HWC0(out_backprop_ori_shape, data_format, dtype)
        dx_shape = _shape_to_NC1HWC0(input_size, data_format, dtype)
        x = {
            "shape": [4],
            "format": "NC1HWC0",
            "ori_shape": [4],
            "ori_format": data_format,
            "dtype": dtype,
            "range": _get_range_from_shape(dx_shape, dynamic_dim)
        }
        filter = {
            "shape": filter_shape,
            "ori_shape": filter_ori_shape,
            "ori_format": filter_format,
            "format": "C1HWNCoC0",
            "dtype": dtype,
            "range": _get_range_from_shape(filter_shape)
        }
        out_backprop = {
            "shape": _trans_dynamic_shape(out_backprop_shape, "NC1HWC0", dynamic_dim, True),
            "format": "NC1HWC0",
            "ori_shape": _trans_dynamic_shape(out_backprop_ori_shape, data_format, dynamic_dim, True),
            "ori_format": data_format,
            "dtype": dtype,
            "range": _get_range_from_shape(out_backprop_shape, dynamic_dim)
        }
        dx = {
            "shape": _trans_dynamic_shape(dx_shape, "NC1HWC0", dynamic_dim),
            "format": "NC1HWC0",
            "ori_shape": _trans_dynamic_shape(input_size, data_format, dynamic_dim),
            "ori_format": data_format,
            "dtype": dtype,
            "range": _get_range_from_shape(dx_shape, dynamic_dim)
        }
    stride_h, stride_w = strides
    strides = [1, stride_h, stride_w, 1] if data_format == "NHWC" else [1, 1, stride_h, stride_w]

    kernel_name = _get_kernel_name(filter_ori_shape, out_backprop_ori_shape, input_size, strides, pads)

    print(x, filter, out_backprop, dx)
    return {
        "params": [x, filter, out_backprop, dx, strides, pads, dilations, group, data_format],
        "case_name": kernel_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }


for case in dynamic_conv2d_bp_input_op_testcase:
    ut_case.add_case(["Ascend910A"], _gen_trans_data_case(case))


def test_conv2d_backprop_input_fuzz_build_generalization_general(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': [(16, 32), (3, 3), (8, 16), (8, 16), (16, 16)],
            'ori_range': [(16, 32), (33, 33), (8, 16), (8, 16)]
        }, {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_general']
    conv2d_backprop_input_generalization(*input_list)


ut_case.add_cust_test_func(test_func=test_conv2d_backprop_input_fuzz_build_generalization_general)


def test_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed(test_arg):
    from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (1, 2, 10, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (50, 1, 26, 2888, 16),
            'ori_shape': (50, 2, 26, 2888),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': [(32, 64), (1, 1), (16, 32), (1024, 4096), (16, 16)],
            'ori_range': [(32, 64), (2, 2), (16, 32), (1024, 4096)]
        }, {
            'shape': (50, 1, 35, 2896, 16),
            'ori_shape': (50, 2, 35, 2896),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_range_max_fixed']
    conv2d_backprop_input_generalization(*input_list)


ut_case.add_cust_test_func(test_func=test_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed)

def test_get_op_support_info_dynamic_dx_0(test_arg):
    y = {"shape": (-1, 4, -1, -1, 16), 'ori_shape': (-1, -1, -1, 64),
         "ori_format": "NHWC", "format": "NC1HWC0", "dtype": "float16",
         "range": ((2, 4), (4, 4), (4, 8), (4, 8), (16, 16))
        }
    out_backprop = {"shape":  (-1, 4, -1, -1, 16), 'ori_shape':(-1, -1, -1, 64),
                    "ori_format": "NHWC", "format": "NC1HWC0", "dtype": "float16",
                    "range": ((2, 4), (4, 4), (4, 8), (4, 8), (16, 16))
                   }
    filter = {"shape":  (36, 4, 16, 16), 'ori_shape':(3, 3, 64, 64),
              "ori_format": "NHWC", "format": "FRATAL_NZ", "dtype": "float16",
              "range": ((36, 36), (4, 4), (16, 16), (16, 16))
             }
    input_size =  (-1, -1, -1, 64)
    get_op_support_info(input_size, filter, out_backprop, y, (1, 1, 1, 1), (0, 0, 0, 0))
def test_get_op_support_info_dynamic_dx_1(test_arg):
    y = {"shape": (-1, 4, -1, -1, 16), 'ori_shape': (-1, -1, -1, 64),
         "ori_format": "NHWC", "format": "NC1HWC0", "dtype": "float16",
         "range": ((2, 4), (4, 4), (4, 8), (4, 8), (16, 16))
        }
    out_backprop = {"shape":  (-1, 4, -1, -1, 16), 'ori_shape':(-1, -1, -1, 64),
                    "ori_format": "NHWC", "format": "NC1HWC0", "dtype": "float16",
                    "range": ((2, 4), (4, 4), (4, 8), (4, 8), (16, 16))
                   }
    filter = {"shape":  (4, 4, 16, 16), 'ori_shape':(1, 1, 64, 64),
              "ori_format": "NHWC", "format": "FRATAL_NZ", "dtype": "float16",
              "range": ((4, 4), (4, 4), (16, 16), (16, 16))
             }
    input_size =  (-1, -1, -1, 64)
    get_op_support_info(input_size, filter, out_backprop, y, (1, 1, 1, 1), (0, 0, 0, 0))

ut_case.add_cust_test_func(test_func=test_get_op_support_info_dynamic_dx_0)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_dynamic_dx_1)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
