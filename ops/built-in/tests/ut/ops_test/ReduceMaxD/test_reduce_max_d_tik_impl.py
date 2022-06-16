#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ReduceMaxD", "impl.reduce_max_d_tik", "reduce_max_d_tik")

def gen_reduce_max_d_case(shape_x, range_x, dtype_val, format,
                          ori_shape_x, axes, keepdims,
                          kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       axes],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_reduce_max_d_case((1,2,3),
                                               [(1,1),(2,2),(3,3)],
                                               "float32", "ND",
                                               (1,2,3), 1, True,
                                               "reduce_max_d_fp32_ND",
                                               "success"))
def test_import_const(_):
    import sys
    import importlib
    importlib.reload(sys.modules.get("impl.constant_util_v1"))

ut_case.add_cust_test_func(test_func=test_import_const)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
