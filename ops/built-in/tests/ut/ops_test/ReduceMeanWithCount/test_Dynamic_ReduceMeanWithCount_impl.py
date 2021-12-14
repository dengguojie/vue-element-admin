#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ReduceMeanWithCount", "impl.dynamic.reduce_mean_with_count", "reduce_mean_with_count")

def gen_dynamic_reduce_mean_with_count_case(shape_x, range_x, ori_shape_x,
                                            dtype_val, format,
                                            axes, keep_dims,
                                            kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val, "range": range_x, "format": format, "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_x, "dtype": dtype_val, "range": range_x, "format": format, "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_x, "dtype": dtype_val, "range": range_x, "format": format, "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_x, "dtype": dtype_val, "range": range_x, "format": format, "ori_shape": ori_shape_x, "ori_format": format},
                       axes, keep_dims],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_reduce_mean_with_count_case((-1,3,-1,7), [(1,None), (3,3), (1,None), (7,7)], (-1,3,-1,7),
                                                         "float16", "ND", [0,3], False, "dynamic_reduce_mean_with_count_4D_fp16_ND_0_3_False", "success"))
ut_case.add_case("all",
                 gen_dynamic_reduce_mean_with_count_case((-1,3,-1,7), [(1,None), (3,3), (1,None), (7,7)], (-1,3,-1,7),
                                                         "float32", "ND", [0,3], False, "dynamic_reduce_mean_with_count_4D_fp32_ND_0_3_False", "success"))
ut_case.add_case("all",
                 gen_dynamic_reduce_mean_with_count_case((-1,3,-1,7), [(1,None), (3,3), (1,None), (7,7)], (-1,3,-1,7),
                                                         "float16", "ND", [0,3], True, "dynamic_reduce_mean_with_count_4D_fp16_ND_0_3_True", "success"))
ut_case.add_case("all",
                 gen_dynamic_reduce_mean_with_count_case((-1,3,-1,7), [(1,None), (3,3), (1,None), (7,7)], (-1,3,-1,7),
                                                         "float32", "ND", [0,3], True, "dynamic_reduce_mean_with_count_4D_fp32_ND_0_3_True", "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910")
    ut_case.run("Ascend710")
    ut_case.run("Ascend310")
