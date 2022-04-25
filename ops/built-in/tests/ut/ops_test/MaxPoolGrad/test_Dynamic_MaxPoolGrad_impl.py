#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import tbe
from unittest.mock import MagicMock
from unittest.mock import patch

ut_case = OpUT("MaxPoolGrad", "impl.dynamic.max_pool_grad", "max_pool_grad")


def gen_dynamic_maxpoolgrad_case(shape_x, shape_y, ori_shape_x, ori_shape_y, range_x, range_y, format, ori_format, dtype_val, ksize, strides, padding, data_format, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_val, "ori_shape": ori_shape_x, "ori_format": ori_format, "format": format, "range": range_x},
                       {"shape": shape_y, "dtype": dtype_val, "ori_shape": ori_shape_y, "ori_format": ori_format, "format": format, "range": range_y},
                       {"shape": shape_y, "dtype": dtype_val, "ori_shape": ori_shape_y, "ori_format": ori_format, "format": format, "range": range_y},
                       {"shape": shape_x, "dtype": dtype_val, "ori_shape": ori_shape_x, "ori_format": ori_format, "format": format, "range": range_x},
                       ksize, strides, padding, data_format,],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("Ascend910A",
                 gen_dynamic_maxpoolgrad_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                           ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                           "NC1HWC0","NHWC","float16",[1,1,1,1],[1,1,1,1],"SAME","NHWC","max_poolgrad_case", "success"))

ut_case.add_case("Ascend910A",
                 gen_dynamic_maxpoolgrad_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[1,2,2,1],[1,3,3,1],"VALID","NHWC","max_poolgrad01_case", "success"))


vals = {("tik.load3dv1",): False}
def side_effects(*args):
    return vals[args]
with patch("te.platform.cce_conf.api_check_support", MagicMock(side_effect=side_effects)):
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")

with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")

