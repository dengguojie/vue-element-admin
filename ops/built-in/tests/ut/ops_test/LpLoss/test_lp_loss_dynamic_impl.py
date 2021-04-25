#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("LpLoss", "impl.dynamic.lp_loss", "lp_loss")

def gen_dynamic_lp_loss_case(_shape, _range, _dtype, _case_name, reduction, _format="ND"):

    return {"params": [{"shape": _shape, "dtype": _dtype, "range": _range, "format": _format, "ori_shape": _shape, "ori_format": _format},
                       {"shape": _shape, "dtype": _dtype, "range": _range, "format": _format, "ori_shape": _shape, "ori_format": _format},
                       {"shape": _shape, "dtype": _dtype, "range": _range, "format": _format, "ori_shape": _shape, "ori_format": _format},
                        1, reduction],
            "case_name": _case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True
            }

case1 = gen_dynamic_lp_loss_case((-1,3,-1,2), [(1,None), (3,3), (1,None), (2,2)], "float32", "dynamiclploss", "sum", "ND")
case2 = gen_dynamic_lp_loss_case((-1,3,-1,2), [(1,None), (3,3), (1,None), (2,2)], "float32", "dynamiclploss", "none", "ND")

ut_case.add_case("all", case1)
ut_case.add_case("all", case2)

if __name__ == "__main__":
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)
    
    
