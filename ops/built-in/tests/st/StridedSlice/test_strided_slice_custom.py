#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.strided_slice import strided_slice
from impl.strided_slice import strided_slice as strided_slice_known
from op_test_frame.ut import OpUT
ut_case = OpUT("StridedSlice", "impl.dynamic.strided_slice", "strided_slice")


# 'pylint: disable=unused-argument
def test_strided_slice(test_arg):
    from tbe.common.context import op_context
    with op_context.OpContext("dynamic"):
        input_list = [
            {
                'shape': (-2,),
                'ori_shape': (-2,),
                'ori_format': 'ND',
                'format': 'ND',
                'dtype': 'bool'
            }, 
            {
                'shape': (-2,),
                'ori_shape': (-2,),
                'ori_format': 'ND',
                'format': 'ND',
                'dtype': 'int32'
            },
            {
                'shape': (-2,),
                'ori_shape': (-2,),
                'ori_format': 'ND',
                'format': 'ND',
                'dtype': 'int32'
            },
            {
                'shape': (-2,),
                'ori_shape': (-2,),
                'ori_format': 'ND',
                'format': 'ND',
                'dtype': 'int32'
            },
            {
                'shape': (-2,),
                'ori_shape': (-2,),
                'ori_format': 'ND',
                'format': 'ND',
                'dtype': 'bool'
            }]
        strided_slice(*input_list)

ut_case.add_cust_test_func(test_func=test_strided_slice)

# 'pylint: disable=unused-argument
def test_canonical(test_arg):
    """
    test for canonical
    """
    strided_slice_known({"shape": (1, 5, 5, 5, 424, 35), "dtype": "float16", "format": "ND",
                        "ori_shape": (1, 5, 5, 5, 424, 35), "ori_format": "ND"},
                        {"shape": (6,), "dtype": "int32", "format": "ND",
                        "ori_shape": (6,), "ori_format": "ND"},
                        {"shape": (6,), "dtype": "int32", "format": "ND",
                        "ori_shape": (6,), "ori_format": "ND"},
                        {"shape": (6,), "dtype": "int32", "format": "ND",
                        "ori_shape": (6,), "ori_format": "ND", "const_value": [1, 1, 1, 1, 1, 1]},
                        {"shape": (1, 2, 4, 4, 190, 5), "dtype": "float16", "format": "ND",
                        "ori_shape": (1, 2, 4, 4, 190, 5), "ori_format": "ND"},
                        0, 0, 0, 0, 0, "strided_slice_known")

ut_case.add_cust_test_func(test_func=test_canonical)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
