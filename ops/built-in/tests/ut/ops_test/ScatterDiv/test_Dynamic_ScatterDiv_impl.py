#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ScatterDiv", "impl.dynamic.scatter_div", "scatter_div")


def gen_dynamic_scatter_div_case(shape_x, shape_y, range_x, range_y, dtype_val, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_val, "ori_shape":shape_x,"ori_format":"ND", "format":"ND","range": range_x},
                       {"shape": shape_y, "dtype": "int32", "ori_shape":shape_y,"ori_format":"ND", "format":"ND","range": range_y},
                       {"shape": shape_x, "dtype": dtype_val, "ori_shape":shape_x,"ori_format":"ND", "format":"ND","range": range_x},
                       {"shape": shape_y, "dtype": dtype_val, "ori_shape":shape_y,"ori_format":"ND", "format":"ND","range": range_y}],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_scatter_div_case((-1,), (1,),
                                       ((1, None),), ((1, 1),),
                                       "float32", "dynamic_scatter_div_case", "success"))

ut_case.add_case("all",
                 gen_dynamic_scatter_div_case((-1,), (1,),
                                       ((1, 1),), ((1, None),),
                                       "float32", "dynamic_scatter_div_case_1", "success"))


if __name__ == '__main__':
    ut_case.run("Ascend910A")
