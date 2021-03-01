#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Div", "impl.dynamic.div", "div")

def gen_dynamic_div_case(shape_x, shape_y, range_x, range_y,
                         dtype_val, format, ori_shape_x, ori_shape_y,
                         kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_y, "dtype": dtype_val,
                        "range": range_y, "format": format,
                        "ori_shape": ori_shape_y, "ori_format": format},
                       {"shape": shape_y, "dtype": dtype_val,
                        "range": range_y, "format": format,
                        "ori_shape": ori_shape_y, "ori_format": format}],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

# (3,7,1,399), (1,9,1)
ut_case.add_case("all",
                 gen_dynamic_div_case((-1,), (-1,),
                                      [(2,7),],
                                      [(2,7),],
                                      "float32", "ND", (-1,), (-1,),
                                      "dynamic_div_fp32_ND", "success"))

if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
