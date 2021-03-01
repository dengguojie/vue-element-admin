#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Atanh", "impl.dynamic.atanh", "atanh")

def gen_dynamic_atanh_case(shape_x, range_x, dtype_val, format, ori_shape_x, kernel_name_val, expect):
    return {"params":[{"shape": shape_x, "dtype": dtype_val,
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
                 gen_dynamic_atanh_case((-1,),
                                       [(1,None)],
                                       "float16", "ND",
                                       (-1,),
                                       "dynamic_atanh_float16_ND",
                                       "success"))
ut_case.add_case("Ascend910A",
                 gen_dynamic_atanh_case((-1,),
                                       [(1,None)],
                                       "int8", "ND",
                                       (-1,),
                                       "dynamic_atanh_int8_ND",
                                       "failed"))
if __name__ == "__main__":
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)