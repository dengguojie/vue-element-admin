#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ScatterNd", "impl.dynamic.scatter_nd", "scatter_nd")


def gen_dynamic_floormod_case(shape_x, shape_y, range_x, range_y, dtype_val, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": "int32", "ori_shape":shape_x,"ori_format":"ND", "format":"ND","range": range_x},
                       {"shape": shape_y, "dtype": "float32", "ori_shape":shape_y,"ori_format":"ND", "format":"ND","range": range_y},
                       {"shape": shape_x, "dtype": "int32", "ori_shape":shape_x,"ori_format":"ND", "format":"ND","range": range_x},
                       {"shape": shape_y, "dtype": "float32", "ori_shape":shape_y,"ori_format":"ND", "format":"ND","range": range_y}],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_floormod_case((-1,), (1,),
                                           ((1, None),), ((1, 1),),
                                           "float32", "dynamic_scatter_nd_case", "success"))

from impl.dynamic.scatter_nd import check_supported
def test_check_support(test_arg):
    check_supported({"shape": (-1,32,1), "dtype": "float16", "format": "ND", "ori_shape": (-1,32,1),"ori_format": "ND", "param_type": "indice"},
    {"shape": (1,32,100), "dtype": "float16", "format": "ND", "ori_shape": (1,32,100),"ori_format": "ND", "param_type": "x"},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,),"ori_format": "ND", "param_type": "shape","const_value":(17, 28, 100)},
    {"shape": (17, 28, 100), "dtype": "float16", "format": "ND", "ori_shape": (17, 28, 100),"ori_format": "ND", "param_type": "y"})
    check_supported({"shape": (1,32,1), "dtype": "float16", "format": "ND", "ori_shape": (1,32,1),"ori_format": "ND", "param_type": "indice"},
    {"shape": (1,32,1), "dtype": "float16", "format": "ND", "ori_shape": (1,32,1),"ori_format": "ND", "param_type": "x"},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,),"ori_format": "ND", "param_type": "shape","const_value":(17, 28, 1)},
    {"shape": (17, 28, 1), "dtype": "float16", "format": "ND", "ori_shape": (17, 28, 1),"ori_format": "ND", "param_type": "y"})
    check_supported({"shape": (1,32,2), "dtype": "float16", "format": "ND", "ori_shape": (1,32,2),"ori_format": "ND", "param_type": "indice"},
    {"shape": (1,32,2), "dtype": "float16", "format": "ND", "ori_shape": (1,32,2),"ori_format": "ND", "param_type": "x"},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,),"ori_format": "ND", "param_type": "shape","const_value":(17, 28, 2)},
    {"shape": (17, 28, 2), "dtype": "float16", "format": "ND", "ori_shape": (17, 28, 2),"ori_format": "ND", "param_type": "y"})

ut_case.add_cust_test_func(test_func=test_check_support)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
