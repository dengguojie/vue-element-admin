#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Sqrt", "impl.dynamic.sqrt", "sqrt")

def gen_dynamic_sqrt_case(shape_x, range_x, dtype_val, format,
                          ori_shape_x, kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format}],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_sqrt_case((-1,),
                                       [(1,None)],
                                       "float32", "ND",
                                       (-1,),
                                       "dynamic_sqrt_float32_ND",
                                       "success"))
ut_case.add_case("all",
                 gen_dynamic_sqrt_case((-1,),
                                       [(1,None)],
                                       "int8", "ND",
                                       (-1,),
                                       "dynamic_sqrt_int8_ND",
                                       "failed"))
ut_case.add_case("all",{"params":[
                       {"shape": (1, 1), "dtype": "float32", "range": [(1, None), (1, None)],
                        "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                       {"shape": (1, 1), "dtype": "float32", "range": [(1, None), (1, None)],
                        "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"}],
                "addition_params": {"impl_mode": "high_precision"},
                "case_name": "no_high_performance_case",
                "expect": "success",
                "format_expect": [],
                "support_expect": True})

ut_case.add_case("all",{"params":[
                       {"shape": (1, 1), "dtype": "float32", "range": [(1, None), (1, None)],
                        "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                       {"shape": (1, 1), "dtype": "float32", "range": [(1, None), (1, None)],
                        "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"}],
                "addition_params": {"impl_mode": "high_performance"},
                "case_name": "is_high_performance_case",
                "expect": "success",
                "format_expect": [],
                "support_expect": True})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
