#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ReduceSum", "impl.dynamic.reduce_sum", "reduce_sum")

def gen_dynamic_reduce_sum_case(shape_x, range_x, dtype_val, format,
                                  ori_shape_x, axes, keepdims,
                                  kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_x, "dtype": "int32",
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       keepdims],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

# ut_case.add_case("all",
#                  gen_dynamic_reduce_sum_case((-1,3,-1,2),
#                                                [(1,None), (3,3),
#                                                (1,None), (2,2)],
#                                                "float16", "ND",
#                                                (-1,3,-1,2), [0,3], True,
#                                                "dynamic_reduce_sum_fp16_ND",
#                                                "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
