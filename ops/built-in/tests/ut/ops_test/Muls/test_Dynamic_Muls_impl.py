#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Muls", "impl.dynamic.muls", "muls")

def gen_dynamic_muls_case(shape_x, range_x, dtype_val, value, format,
                          ori_shape_x, kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                        value],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_muls_case(shape_x=(-1,),
                                       range_x=[(1,None)],
                                       dtype_val="float32",
                                       value=2.0,
                                       format="ND",
                                       ori_shape_x=(-1,),
                                       kernel_name_val="dynamic_muls_float32_ND",
                                       expect="success"))
ut_case.add_case("all",
                 gen_dynamic_muls_case((-1,),
                                       [(1,None)],
                                       "int8", 
                                       -3,
                                       "ND",
                                       (-1,),
                                       "dynamic_sqrt_int8_ND",
                                       "failed"))

if __name__ == '__main__':
    import te
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)
