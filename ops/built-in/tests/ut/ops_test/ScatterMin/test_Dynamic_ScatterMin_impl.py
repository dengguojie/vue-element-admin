#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ScatterMin", "impl.dynamic.scatter_min", "scatter_min")


def gen_dynamic_scatter_min_case(shape_var, shape_indices, shape_updates, range_var, range_indices, range_updates, dtype, kernel_name_val, expect):
    return {"params":
    [{"shape": shape_var, "dtype": dtype, "ori_shape":shape_var,"ori_format":"ND", "format":"ND","range": range_var},
     {"shape": shape_indices, "dtype": "int32", "ori_shape":shape_indices,"ori_format":"ND", "format":"ND","range": range_indices},
     {"shape": shape_updates, "dtype": dtype, "ori_shape":shape_updates,"ori_format":"ND", "format":"ND","range": range_updates},
     {"shape": shape_var, "dtype": dtype, "ori_shape":shape_var,"ori_format":"ND", "format":"ND","range": range_var}],
    "case_name": kernel_name_val,
    "expect": expect,
    "format_expect": [],
    "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_scatter_min_case((-1,34,23,-1), (1,), (-1,34,23,-1),
                ((23, None),(34,34),(23,23),(12,80)), ((1, None),), ((23, None),(34,34),(23,23),(12,80)),
                "float32", "dynamic_scatter_min_case_1", "success"))


if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)
