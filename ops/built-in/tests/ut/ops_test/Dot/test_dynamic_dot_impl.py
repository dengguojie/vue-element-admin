"""
Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

DynamicDot ut case
"""
# -*- coding:utf-8 -*-
import numpy as np
from op_test_frame.ut  import OpUT

ut_case = OpUT("Dot", "impl.dynamic.dot", "dot")


def calc_expect_func(input_x, input_y):
    res = np.dot(input_x["value"], input_y["value"])
    result = res.reshape((1, ))
    return [result, ]


def gen_dynamic_dot_case(shape_x,shape_y,range_x,dtype_val,format_x,format_y,ori_shape_x,
                         ori_shape_y,ori_format_x,ori_format_y,kernel_name_val,expect,op_imply_type):
    return {"params": [{"dtype": dtype_val, "format": format_x, "ori_format": ori_format_x, "ori_shape": ori_shape_x, "shape": shape_x,
                        "range": range_x},
                       {"dtype":dtype_val, "format": format_x, "ori_format": ori_format_x, "ori_shape": ori_shape_x, "shape": shape_x,
                        "range": range_x},
                       {"dtype": dtype_val, "format": format_y, "ori_format": ori_format_y, "ori_shape": ori_shape_y, "shape": shape_y
                        }],
                        "case_name":kernel_name_val,
                        "expect":expect,
                        "op_imply_type":op_imply_type,
                        "format_expect":["ND"],
						"calc_expect_func":calc_expect_func,
                        "support_expect":True
                        }

ut_case.add_case("all",gen_dynamic_dot_case((-1,),(1,),[(1,100)],"float16","ND","ND",
                                            (32,),(1,),"ND","ND","dynamic_dot_fp32","success","dynamic"))        
                                            
if __name__ == "__main__":
    ut_case.run(["Ascend910","Ascend710","Ascend610","Ascend310"])