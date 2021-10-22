#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from impl.util.platform_adapter import tbe_context
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import os
import time

#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(linewidth=10000)

ut_case = OpUT("Transpose", "impl.dynamic.transpose", "transpose")

def test_op_check_supported_for_fe(test_arg):
    from impl.transpose import check_supported
    input_x = {'ori_shape': (-1, -1), 'shape': (2, 3), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    perm = {'ori_shape': (-1,), 'shape': (2,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (-1, -1), 'shape': (3, 2), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    check_supported(input_x, perm, output_y)

ut_case.add_cust_test_func(test_func=test_op_check_supported_for_fe)

if __name__ == '__main__':
    simulator_lib_path = "/usr/local/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend910A", "Ascend920A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)

