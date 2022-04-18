#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ReduceMinD", "impl.dynamic.reduce_min_d", "reduce_min_d")

def gen_dynamic_reduce_min_d_case(shape_x, range_x, dtype_val, format,
                                  ori_shape_x, axes, keepdims,
                                  kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       axes, keepdims],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_reduce_min_d_case((-1,-1,1),
                                               [(1,100),(1,100),(1,1)],
                                               "float16", "ND",
                                               (-1,-1,1), [1,], True,
                                               "dynamic_reduce_min_d_fp16_ND",
                                               "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
