#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = OpUT("Select", "impl.select", "op_select_format")

def gen_select_case(shape_var, dtype, expect, case_name_val, input_format, bool_dtype="int8"):
    return {"params": [{"shape": shape_var, "dtype": bool_dtype, "ori_shape": shape_var, "ori_format": input_format, "format": input_format},
                       {"shape": shape_var, "dtype": dtype, "ori_shape": shape_var, "ori_format": input_format, "format": input_format},
                       {"shape": shape_var, "dtype": dtype, "ori_shape": shape_var, "ori_format": input_format, "format": input_format},
                       {"shape": shape_var, "dtype": dtype, "ori_shape": shape_var, "ori_format": input_format, "format": input_format}],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

case1 = gen_select_case((16, 16, 16, 16, 16), "float16", "success", "select_op_select_format_6hd_1", "NDHWC")
case1 = gen_select_case((16, 16, 16, 16), "float16", "success", "select_op_select_format_5hd_1", "NCHW")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
