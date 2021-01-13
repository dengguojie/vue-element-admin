#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import te

ut_case = OpUT("ReciprocalGrad", "impl.dynamic.reciprocal_grad", "reciprocal_grad")

def gen_dynamic_reciprocal_grad_case(shape_x, shape_y,
                                     range_x, range_y,
                                     dtype_val, format_val, 
                                     ori_shape_x, ori_shape_y,
                                     kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format_val,
                        "ori_shape": ori_shape_x, "ori_format": format_val},
                       {"shape": shape_y, "dtype": dtype_val,
                        "range": range_y, "format": format_val,
                        "ori_shape": ori_shape_y, "ori_format": format_val},
                       {"shape": shape_y, "dtype": dtype_val,
                        "range": range_y, "format": format_val,
                        "ori_shape": ori_shape_y, "ori_format": format_val}],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

# (3,7,1,399), (1,9,1)
ut_case.add_case("all",
                 gen_dynamic_reciprocal_grad_case((-1,), (-1,),
                                                  [(2, 10)], [(2, 10)],
                                                  "float32", "ND", 
                                                  (-1,), (-1,),
                                                  "dynamic_reciprocal_grad_fp32_ND", "success"))

with te.op.dynamic():
    ut_case.run("Ascend910")
