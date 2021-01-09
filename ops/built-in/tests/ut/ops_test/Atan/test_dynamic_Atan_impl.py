#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Atan", "impl.dynamic.atan", "atan")
def gen_dynamic_atan_case(shape_x, range_x, dtype_val, format, ori_shape_x, kernel_name_val, expect):
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
            
ut_case.add_case("all",
                 gen_dynamic_atan_case((-1,),
                                       [(1,None)],
                                       "float32", "ND",
                                       (-1,),
                                       "dynamic_atan_float32_ND",
                                       "success"))
ut_case.add_case("all",
                 gen_dynamic_atan_case((-1,),
                                       [(1,None)],
                                       "int8", "ND",
                                       (-1,),
                                       "dynamic_atan_int8_ND",
                                       "failed"))
if __name__ == "__main__":
    import te
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)