#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.reduce_all import reduce_all
from op_test_frame.ut import OpUT
ut_case = OpUT("ReduceAll", "impl.dynamic.reduce_all", "reduce_all")


def test_reduce_all(test_arg):
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
                'dtype': 'bool'
            }]
        reduce_all(*input_list)

ut_case.add_cust_test_func(test_func=test_reduce_all)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
