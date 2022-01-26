#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.zeros_like import zeros_like
from op_test_frame.ut import OpUT
ut_case = OpUT("ZerosLike", "impl.dynamic.zeros_like", "zeros_like")


def test_zeros_like(test_arg):
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
                'dtype': 'bool'
            }]
        zeros_like(*input_list)

ut_case.add_cust_test_func(test_func=test_zeros_like)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
