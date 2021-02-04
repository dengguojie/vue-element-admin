#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("AccumulateNv2", "impl.accumulate_nv2", "accumulate_nv2")

case1 = {"params": [[{"shape": (1, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3),"ori_format": "NCHW"},
                     {"shape": (1, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3),"ori_format": "NCHW"}],
                    {"shape": (1, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 3),"ori_format": "NCHW"},
                    2],
         "case_name": "accumulate_nv2_comp_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [[{"shape": (2, 4, 6, 10), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 4, 6, 10),"ori_format": "NCHW"},
                     {"shape": (2, 4, 6, 10), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 4, 6, 10),"ori_format": "NCHW"}],
                    {"shape": (2, 4, 6, 10), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 4, 6, 10),"ori_format": "NCHW"},
                    2],
         "case_name": "accumulate_nv2_comp_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [[{"shape": (2, 1, 6, 10), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 1, 6, 10),"ori_format": "NCHW"},
                     {"shape": (2, 4, 1, 10), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 4, 1, 10),"ori_format": "NCHW"}],
                    {"shape": (2, 4, 6, 10), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 4, 6, 10),"ori_format": "NCHW"},
                    2],
         "case_name": "accumulate_nv2_comp_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310"], case2)
# ut_case.add_case(["Ascend310"], case3)

def get_broad_cast_shape(input_dic_list):
    input_shape_num = len(input_dic_list)
    input_shape_list = []
    for input_tensor in input_dic_list:
        t_shape = list(input_tensor.get("shape"))
        input_shape_list.append(t_shape)
    
    max_dim = max([len(s) for s in input_shape_list])
    for i in range(input_shape_num):
        shape = input_shape_list[i]
        input_shape_list[i] = [1] * (max_dim - len(shape)) + shape
    
    output_shape = [0] * max_dim
    for dim in range(max_dim):
        broad_cast_value = input_shape_list[0][dim]
        for num in range(1, input_shape_num):
            if broad_cast_value == 1:
                if input_shape_list[num][dim] != 1:
                    broad_cast_value = input_shape_list[num][dim]
            else:
                if input_shape_list[num][dim] != 1 and input_shape_list[num][dim] != broad_cast_value:
                    return -1
        output_shape[dim] = broad_cast_value

    return output_shape

def calc_expect_func(x_list, res, num):
    if len(x_list) != num:
        print("len x_list is not same with num.")
        return None

    broad_cast_shape = get_broad_cast_shape(x_list)
    print("broad_cast_shape is:", broad_cast_shape)
    date_type = x_list[0].get("dtype")
    if broad_cast_shape != list(res.get("shape")):
        print("broad_cast_shape is not same with res shape")
        return None

    res_value = np.zeros(broad_cast_shape, dtype = date_type)

    for x in x_list:
        x_value = x.get("value")
        res_value = np.add(x_value, res_value)

    return res_value

ut_case.add_precision_case("Ascend310", 
                           {"params": [[{"shape": (2, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND", "param_type": "input"},
                                        {"shape": (2, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND", "param_type": "input"}],
                                       {"shape": (2, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND", "param_type": "output"},
                                       2],
                            "case_name": "accumulate_nv2_prec_1",
                            "calc_expect_func": calc_expect_func,
                            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
ut_case.add_precision_case("Ascend310", 
                           {"params": [[{"shape": (2, 4, 6, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 4, 6, 8),"ori_format": "NCHW", "param_type": "input"},
                                        {"shape": (2, 4, 6, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 4, 6, 8),"ori_format": "NCHW", "param_type": "input"},
                                        {"shape": (2, 4, 6, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 4, 6, 8),"ori_format": "NCHW", "param_type": "input"}],
                                       {"shape": (2, 4, 6, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 4, 6, 8),"ori_format": "NCHW", "param_type": "output"},
                                       3],
                            "case_name": "accumulate_nv2_prec_2",
                            "calc_expect_func": calc_expect_func,
                            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})
# ut_case.add_precision_case("Ascend310", 
#                            {"params": [[{"shape": (2, 1, 1, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 1, 1, 1),"ori_format": "NCHW", "param_type": "input"},
#                                         {"shape": (2, 4, 1, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 4, 1, 8),"ori_format": "NCHW", "param_type": "input"},
#                                         {"shape": (2, 1, 6, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 1, 6, 8),"ori_format": "NCHW", "param_type": "input"}],
#                                        {"shape": (2, 4, 6, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 4, 6, 8),"ori_format": "NCHW", "param_type": "output"},
#                                        3],
#                             "case_name": "accumulate_nv2_prec_3",
#                             "calc_expect_func": calc_expect_func,
#                             "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})

if __name__ == '__main__':
    ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
