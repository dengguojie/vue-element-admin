#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("DataFormatDimMap", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}],
         "case_name": "data_format_dim_map_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1,2), "dtype": "int32", "format": "NCHW", "ori_shape": (1,2),"ori_format": "NCHW"},
                    {"shape": (1,2), "dtype": "int32", "format": "NCHW", "ori_shape": (1,2),"ori_format": "NCHW"}],
         "case_name": "data_format_dim_map_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

def calc_expect_func(x, y, src, dst):
    input_Arr = x['value']
    shape = x['shape']
    result_mod = np.ndarray.__mod__(input_Arr, 4)
    ind = []
    for i in range(0, len(src)):
        for j in range(0, len(dst)):
            if src[i] == dst[j]:
                ind.insert(i, j)
    zero = np.zeros(shape)
    ind_0_bc = zero + ind[0]
    ind_1_bc = zero + ind[1]
    ind_2_bc = zero + ind[2]
    ind_3_bc = zero + ind[3]
    is_zero = (result_mod == 0)
    is_one = (result_mod == 1)
    is_two = (result_mod == 2)
    result = np.where(is_zero, ind_0_bc, np.where(is_one, ind_1_bc, np.where(is_two, ind_2_bc, ind_3_bc)))
    return result

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "int32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input","value_range":[-4,4]},
                                              {"shape": (1, 1), "dtype": "int32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              "NHWC", "NCHW"],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (3, 16, 32), "dtype": "int32", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "input","value_range":[-4,4]},
                                              {"shape": (3, 16, 32), "dtype": "int32", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              "NHWC", "NCHW"],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 3, 100, 16), "dtype": "int32", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input","value_range":[-4,4]},
                                              {"shape": (1, 3, 100, 16), "dtype": "int32", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "output"},
                                              "NHWC", "NCHW"],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

