#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Relu6D", "impl.dynamic.relu6_d", "relu6_d")

case1 = {"params": [{"shape": (-2,), "dtype": "float16", "format": "ND", "ori_shape": (-2,),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    {"shape": (-2,), "dtype": "float16", "format": "ND", "ori_shape": (-2,),
                     "ori_format": "ND", "range": [(1, 1), (1, None)]},
                    1.0],
         "case_name": "relu6_d_float16_1",
         "expect": "success"}

ut_case.add_case(["Ascend910A"], case1)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
