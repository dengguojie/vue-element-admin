#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np

ut_case = OpUT("Celu",None,None)

def calc_expect_func(input_x,output_z):

    indtype = input_x["dtype"]
    if indtype == "float32":
        outdtype = np.float32
    if indtype == "float16":
        outdtype = np.float16

    x = input_x["value"].astype(np.float32)
    s = 1-np.exp(x)
    res = (abs(x)+x)/2 - (abs(s)+s)/2
    res = res.astype(outdtype)
    return res

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "output"},],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"},],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "output"},],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "output"},],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"},],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "output"},],
                                   "calc_expect_func": calc_expect_func
                                   })

case_error = {"params": [{"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                         {"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                         1.0, 0, 1.0],
                         "case_name": "case_error",
             "expect": RuntimeError,
             "format_expect": [],
             "support_expect": True}
ut_case.add_case(["Ascend910A"], case_error)