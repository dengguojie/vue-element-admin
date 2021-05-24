#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.deconvolution import get_op_support_info
from op_test_frame.ut import OpUT

ut_case = OpUT("Deconvolution", "impl.dynamic.deconvolution",
               "deconvolution")

dynamic_deconvolution_testcase = [ 
    ((192, 192, 5, 5), (1, 192, -1, 28), (1, 192, -1, 28), (2, 1), (2, 2, 2, 2), "NCHW", 1, [0, 2], "success"),
    ((16, 16, 3, 3), (2, 32, 5, 5), (2, 16, 5, 5), (1, 1), (1, 1, 1, 1), "NCHW", 1, [0, 1], "success"),
    ((16, 16, 3, 3), (2, 32, 5, 5), (2, 16, 5, 5), (1, 1), (1, 1, 1, 1), "NCHW", 1, [1, 3], "success"),
    ((16, 16, 3, 3), (2, 32, 1, 1), (2, 16, 4, 3), (2, 1), (0, 0, 0, 0), "NCHW", 1, [1, 2], "success"),
    ((96, 96, 3, 3), (-1, 96, 2, -1), (-1, 96, 2, -1), (1, 2), (1, 1, 1, 1), "NCHW", 1, [0, 3], "success"),
    ((256, 256, 3, 3), (2, 256, 34, 32), (2, 256, 36, 34), (1, 1), (0, 0, 0, 0), "NCHW", 1, [0, 2, 3], "success"),
    ((240, 240, 5, 5), (2, 240, 1, 1), (2, 240, 2, 1), (2, 1), (2, 2, 2, 2), "NCHW", 1, [0, 2, 3], "success"),   
    ((16, 16, 3, 3), (2, 32, 5, 5), (2, 16, 5, 5), (1, 1), (1, 1, 1, 1), "NCHW", 1, [0, 1, 2, 3], "success"),
    ((32, 32, 3, 3), (32, 32, 7, 1), (32, 32, 7, 2), (1, 2), (1, 1, 1, 1), "NCHW", 1, [0], "success"),
    ((32, 32, 1, 1), (1, 32, 28, 55), (1, 32, 55, 55), (2, 1), (0, 0, 0, 0), "NCHW", 1, [2, 3], "success"), 
    ((16, 16, 3, 3), [-2], (1, 16, 3, 3), (1, 1), (0, 0, 0, 0), "NCHW", 1, [0, 1, 2, 3], "success"),
    ((10, 64, 7, 6), [-2], (1, 64, 1, 1), (1, 1), (1, 1, 1, 1), "NCHW", 1, [0, 1, 2, 3], "success"),
    
    ((16, 16, 3, 3), (2, 32, 5, 5), (2, 16, 5, 5), (1, 1), (1, 1, 1, 1), "NCHW", 1, [0, 2, 3], RuntimeError),
    ((16, 16, 3, 3), (2, 32, 5, 5), (2, 16, 5, 5), (1, 1), (1, 1, 1, 1), "NCHW", 1, [1], RuntimeError),
]

def _get_kernel_name(filter_shape, dy_shape, x_shape, strides, pads):
    padding = "SAME" if -1 in pads else "VALID"
    if dy_shape == [-2]:
        dy_shape_info = "neg_2"
    else:
        dy_shape_info = '_'.join(map(str, dy_shape))
    kernel_name = 'dynamic_deconvolution_' + '_'.join(map(str, filter_shape)) + '_' + dy_shape_info + '_' + '_'.join(
        map(str, x_shape)) + '_' + '_'.join(map(str, strides)) + "_" + padding
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
    if out_backprop_ori_shape == [-2]:
        input_size_op = [-1, input_size[data_format.index("C")], -1, -1]
        filter = {'shape': _shape_to_C1HWNCoC0(filter_ori_shape, "NCHW", dtype),
                  'ori_shape': filter_ori_shape, 'ori_format': 'NCHW',
                  'format': 'C1HWNCoC0', 'dtype': 'float16', 'range': None}
        out_backprop = {'shape': out_backprop_ori_shape, 'format': 'NC1HWC0', 'ori_shape': out_backprop_ori_shape,
                        'ori_format': "NCHW", 'dtype': 'float16', 'range': None}
        dx = {'shape': _shape_to_NC1HWC0(input_size_op, data_format, dtype), 'format': 'NC1HWC0',
              'ori_shape': input_size_op, 'ori_format': "NCHW", 'dtype': 'float16', 'range': None}
    else:
        filter_shape = _shape_to_C1HWNCoC0(filter_ori_shape, "NCHW", dtype),
        out_backprop_shape = _shape_to_NC1HWC0(out_backprop_ori_shape, data_format, dtype)
        dx_shape = _shape_to_NC1HWC0(input_size, data_format, dtype)

        filter = {
            "shape": filter_shape,
            "ori_shape": filter_ori_shape,
            "ori_format": "NCHW",
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
    bias = None
    offset_w = None
    kernel_name = _get_kernel_name(filter_ori_shape, out_backprop_ori_shape, input_size, strides, pads)

    return {
        "params": [out_backprop, filter, bias, offset_w, dx, strides, pads, dilations, group, data_format],
        "case_name": kernel_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }

for case in dynamic_deconvolution_testcase:
    ut_case.add_case(["Ascend910A"], _gen_trans_data_case(case))

def test_deconvolution_fuzz_build_generalization(test_arg):
    from impl.dynamic.deconvolution import deconvolution_generalization
    input_list = [
        {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', 0,
        'deconvolution_fuzz_build_generalization']
    deconvolution_generalization(*input_list)

print("adding conv2d test_deconvolution_fuzz_build_generalization testcase")
ut_case.add_cust_test_func(test_func=test_deconvolution_fuzz_build_generalization)

def test_get_op_support_info_dynamic_deconv(test_arg):
    x = {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format": "NCHW", "shape": (1, 1, -1, -1, 16), "format":"NC1HWC0",
         "range": ((1, 1), (1, 1), (10, 20), (10, 20), (16, 16))}
    weight = {"ori_shape": (16, 16, 1, 1), "dtype": "float16", "ori_format": "NCHW", "shape": (1, 1, 16, 16), "format": "FRACTAL_NZ",
              "range": ((1, 1), (1, 1), (16, 16), (16, 16))}
    bias = None
    y = {"ori_shape": (1, 16, -1, -1), "dtype": "float16", "ori_format": "NCHW",  "shape": (1, 1, -1, -1, 16), "format":"NC1HWC0",
         "range": ((1, 1), (1, 1), (10, 20), (10, 20), (16, 16))}
    get_op_support_info(x, weight, bias, None, y, (1, 1), (0, 0, 0, 0))
ut_case.add_cust_test_func(test_func=test_get_op_support_info_dynamic_deconv)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
