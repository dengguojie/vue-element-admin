#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Sign", "impl.dynamic.sign", "sign")

def gen_dynamic_sign_case(shape_x, range_x, dtype_val, format,
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

ut_case.add_case("Ascend910A",
                 gen_dynamic_sign_case((-1,),
                                       [(1,None)],
                                       "float32", "ND",
                                       (-1,),
                                       "dynamic_sign_float32_ND",
                                       "success"))

if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)
