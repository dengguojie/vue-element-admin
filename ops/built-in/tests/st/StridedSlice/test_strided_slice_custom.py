#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.strided_slice import strided_slice
from op_test_frame.ut import OpUT
ut_case = OpUT("StridedSlice", "impl.dynamic.strided_slice", "strided_slice")


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
if __name__ == '__main__':
    ut_case.run("Ascend910A")