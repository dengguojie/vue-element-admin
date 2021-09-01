#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("LpNormUpdate", "impl.dynamic.lp_norm_update", "lp_norm_update")

def gen_dynamic_lp_norm_update_case(shape_x, range_x, dtype_val, _format,
                                  ori_shape_x, p, kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": _format,
                        "ori_shape": ori_shape_x, "ori_format": _format},
                       {"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format" : _format,
                        "ori_shape": ori_shape_x, "ori_format": _format},
                       p],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_lp_norm_update_case((-1,-1,1),
                                                      [(1,100),(1,100),(1,1)],
                                                      "float16", "ND",
                                                      (-1,-1,1), 2147483647,
                                                      "dynamic_lp_norm_update_fp16_ND_1",
                                                      "success"))
ut_case.add_case(["Ascend910A"],
                 gen_dynamic_lp_norm_update_case((-1,-1,1),
                                                      [(1,100),(1,100),(1,1)],
                                                      "float16", "ND",
                                                      (-1,-1,1), -2147483647,
                                                      "dynamic_lp_norm_update_fp16_ND_2",
                                                      "success"))

ut_case.add_case(["Ascend910A"],
                 gen_dynamic_lp_norm_update_case((-1,-1,1),
                                                      [(1,100),(1,100),(1,1)],
                                                      "float16", "ND",
                                                      (-1,-1,1), 0,
                                                      "dynamic_lp_norm_update_fp16_ND_3",
                                                      "success"))
ut_case.add_case("all",
                 gen_dynamic_lp_norm_update_case((-1,-1,1),
                                                      [(1,100),(1,100),(1,1)],
                                                      "float16", "ND",
                                                      (-1,-1,1), 1,
                                                      "dynamic_lp_norm_update_fp16_ND_4",
                                                      "success"))
ut_case.add_case("all",
                 gen_dynamic_lp_norm_update_case((-1,-1,1),
                                                      [(1,100),(1,100),(1,1)],
                                                      "float16", "ND",
                                                      (-1,-1,1), 2,
                                                      "dynamic_lp_norm_update_fp16_ND_5",
                                                      "success"))
ut_case.add_case("all",
                 gen_dynamic_lp_norm_update_case((-1,-1,1),
                                                      [(1,100),(1,100),(1,1)],
                                                      "float32", "ND",
                                                      (-1,-1,1), 2,
                                                      "dynamic_lp_norm_update_fp32_ND_1",
                                                      "success"))                                                
ut_case.add_case(["Ascend910A"],
                 gen_dynamic_lp_norm_update_case((-1,-1,1),
                                                      [(1,100),(1,100),(1,1)],
                                                      "float32", "ND",
                                                      (-1,-1,1), 3,
                                                      "dynamic_lp_norm_update_fp32_ND_2",
                                                      "success"))
ut_case.add_case(["Ascend910A"],
                 gen_dynamic_lp_norm_update_case((-1,-1,1),
                                                      [(1,100),(1,100),(1,1)],
                                                      "float32", "ND",
                                                      (-1,-1,1), 3,
                                                      "dynamic_lp_norm_update_fp32_ND_3",
                                                      "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
