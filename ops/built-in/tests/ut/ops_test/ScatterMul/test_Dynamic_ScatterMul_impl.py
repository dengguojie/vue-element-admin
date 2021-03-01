#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ScatterMul", "impl.dynamic.scatter_mul", "scatter_mul")


def gen_dynamic_scatter_mul_case(shape_x, shape_y, range_x, range_y, dtype_val, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_val, "ori_shape":shape_x,"ori_format":"ND", "format":"ND","range": range_x},
                       {"shape": shape_y, "dtype": "int32", "ori_shape":shape_y,"ori_format":"ND", "format":"ND","range": range_y},
                       {"shape": shape_x, "dtype": dtype_val, "ori_shape":shape_x,"ori_format":"ND", "format":"ND","range": range_x},
                       {"shape": shape_y, "dtype": dtype_val, "ori_shape":shape_y,"ori_format":"ND", "format":"ND","range": range_y}],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_scatter_mul_case((-1,), (1,),
                                       ((1, None),), ((1, 1),),
                                       "float32", "dynamic_scatter_mul_case", "success"))


if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)
