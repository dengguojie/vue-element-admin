#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ReduceMeanD", "impl.dynamic.reduce_mean_d", "reduce_mean_d")

def gen_dynamic_reduce_mean_d_case(shape_x, shape_y, range_x, range_y, dtype_val, format,
                                   ori_shape_x, ori_shape_y, axes, keepdims,
                                   kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_y, "dtype": dtype_val,
                        "range": range_y, "format": format,
                        "ori_shape": ori_shape_y, "ori_format": format},
                       axes, keepdims],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_reduce_mean_d_case((-1,3,-1,7), (-1,3,-1,7),
                                                [(1,None), (3,3),
                                                 (1,None), (7,7)],
                                                 [(1,None), (3,3),
                                                 (1,None), (7,7)],
                                                "float16", "ND",
                                                (-1,3,-1,7), (-1,3,-1,7), [0,3], True,
                                                "dynamic_reduce_mean_d_fp16_ND",
                                                "success"))
ut_case.add_case("all",
                 gen_dynamic_reduce_mean_d_case((-1,3,-1,7), (-1,3,-1,7),
                                                [(0,None), (3,3),
                                                 (1,None), (7,7)],
                                                 [(1,None), (3,3),
                                                 (1,None), (7,7)],
                                                "float16", "ND",
                                                (-1,3,-1,7), (-1,3,-1,7), [0,3], True,
                                                "dynamic_reduce_mean_d_fp16_ND_1",
                                                "success"))                                                

if __name__ == '__main__':
    ut_case.run("Ascend910A")
