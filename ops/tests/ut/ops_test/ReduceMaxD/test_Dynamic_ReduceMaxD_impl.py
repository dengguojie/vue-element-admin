#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ReduceMaxD", "impl.dynamic.reduce_max_d", "reduce_max_d")

def gen_dynamic_reduce_max_d_case(shape_x, range_x, dtype_val, format,
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
                 gen_dynamic_reduce_max_d_case((-1,-1,1),
                                               [(1,100),(1,100),(1,1)],
                                               "float16", "ND",
                                               (-1,-1,1), [1,], True,
                                               "dynamic_reduce_max_d_fp16_ND",
                                               "success"))

if __name__ == '__main__':
    import te
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)
