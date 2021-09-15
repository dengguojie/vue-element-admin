#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from te import tvm
from op_test_frame.ut import OpUT
from impl.trans_data import *

ut_case = OpUT("TransData", "impl.dynamic.trans_data", "trans_data")

def test_transdata_1(test_arg):
    try:
        input_tensor = tvm.placeholder([100, 1, 112, 16], name='input_tensor', dtype="float16",
                                       attrs={'ori_format': 'NCHW'})
        output_tensor = {'dtype': 'float16', 'format': 'NCHW', 'ori_format': 'NCHW', 'ori_shape': [100, 1, 7, 16], 'shape': [100, 1, 7, 16]}
        trans_data_compute(input_tensor, output_tensor, "NC1HWC0", "NCHW")
    except RuntimeError:
        pass

ut_case.add_cust_test_func(test_func=test_transdata_1)
