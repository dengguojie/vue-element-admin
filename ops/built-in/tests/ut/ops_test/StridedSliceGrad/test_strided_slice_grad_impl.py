#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("StridedSliceGrad", "impl.strided_slice_grad", "strided_slice_grad")

def test_op_check_supported_1(test_arg):
    from impl.strided_slice_grad import check_supported
    shape = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    begin = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    end = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    strides = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16', "const_value": (1, 1, 1, 3)}
    dy = {'ori_shape': (-1, -1, -1, -1), 'shape': (32, 32, 32, 256), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    output = {'ori_shape': (-1, -1, -1, -1), 'shape': (32, 32, 32, 768), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    if check_supported(shape, begin, end, strides, dy, output, new_axis_mask=0, shrink_axis_mask=0) == False:
        raise Exception("Failed to call check_supported in stridedslicegrad.")

ut_case.add_cust_test_func(test_func=test_op_check_supported_1)

if __name__ == '__main__':
    ut_case.run("Ascend910A")