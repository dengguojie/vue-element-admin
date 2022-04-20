#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
from te import tvm
from tbe.dsl import auto_schedule
from impl.fully_connection import fully_connection_compute
from impl.ascend_requant import ascend_requant_compute
from te import platform as cceconf
import te.lang.cce as tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("FullyConnection", None, None)

testcase_list = (
    (
        {'shape': (1, 128, 1, 1, 32), 'dtype': 'int8', 'format': 'NC1HWC0'},
        {'shape': (128, 256, 16, 32), 'dtype': 'int8', 'format': 'FRACTAL_Z '},
        {'shape': (1, 256, 1, 1, 16), 'dtype': 'int32', 'format': 'NC1HWC0', 'ori_format': 'NC1HWC0', 'ori_shape': (1, 256, 1, 1, 16)},
        None,
        {'shape': (1, 1, 64, 16), 'dtype': 'int32', 'format': "FRACTAL_NZ"},
        64*16, False, 1, -128,
        {'shape': (1, 1, 1, 1, 16), 'dtype': 'uint64'}
    ),
)


def get_kernel_name(x, w, b):
    shape_x = x.get('shape')
    shape_w = w.get('shape')
    if b is not None:
        str_bias = 'bias'
    else:
        str_bias = 'no_bias'

    kernel_name_val = "fully_connection_requant_fusion_{}_{}_{}".format(
        "_".join(str(i) for i in shape_x),
        "_".join(str(i) for i in shape_w),
        str_bias)
    return kernel_name_val


def test_fully_connection_requant_fusion(x, w, b, offset_w, y, num_output, transpose, axis, offset_x, deq):
    kernel_name_val = get_kernel_name(x, w, b)
    print('running', kernel_name_val)
    cceconf.te_set_version("Ascend710", "AiCore")
    shape_x = x.get('shape')
    dtype_x = x.get('dtype')
    format_x = x.get('format')
    shape_x_final = (shape_x[0], shape_x[1] * shape_x[2] * shape_x[3] * shape_x[4])

    tensor_x = tvm.placeholder(shape_x_final, dtype=dtype_x,
                               name='tensor_a',
                               attrs={'format': format_x})
    shape_w = w.get('shape')
    dtype_w = w.get('dtype')
    tensor_w = tvm.placeholder(shape_w, dtype=dtype_w, name='tensor_b')

    if b is not None:
        shape_b = b.get('shape')
        dtype_b = b.get('dtype')
        b_size = shape_b[1] * shape_b[4]
        shape_bias = (b_size, )
        tensor_b = tvm.placeholder(shape_bias, dtype=dtype_b, name='tensor_bias', attrs={'ori_shape': shape_bias})
    else:
        tensor_b = None

    tensor_offset_w = None
    res = fully_connection_compute(tensor_x, tensor_w, tensor_b,
                                   tensor_offset_w, y, num_output,
                                   transpose, axis, offset_x)

    shape_deq = deq.get('shape')
    dtype_deq = deq.get('dtype')
    tensor_deq = tvm.placeholder(shape_deq, dtype=dtype_deq, name='tensor_req', attrs={'ori_shape': [1]})
    res = ascend_requant_compute(res, tensor_deq, y, relu_flag=False)

    with tvm.target.cce():
        sch = auto_schedule(res)

    if b is not None:
        tensor_list = [tensor_x, tensor_w, tensor_b, tensor_deq, res]
    else:
        tensor_list = [tensor_x, tensor_w, tensor_deq, res]

    config = {
        'print_ir': False,
        'need_build': True,
        'name': kernel_name_val,
        'tensor_list': tensor_list
    }

    tbe.cce_build_code(sch, config)


def test_fully_connection_requant_fusion_dsl():

    def test_func(test_arg):
        for case in testcase_list:
            x = case[0]
            w = case[1]
            b = case[2]
            offset_w = case[3]
            y = case[4]
            num_output = case[5]
            transpose = case[6]
            axis = case[7]
            offset_x = case[8]
            deq = case[9]
            test_fully_connection_requant_fusion(x, w, b, offset_w, y, num_output, transpose, axis, offset_x, deq)

    return test_func

ut_case.add_cust_test_func(
    ["Ascend710", "Ascend910A"], test_func=test_fully_connection_requant_fusion_dsl()
)

if __name__ == "__main__":
    ut_case.run(["Ascend910A"])
    sys.exit(0)