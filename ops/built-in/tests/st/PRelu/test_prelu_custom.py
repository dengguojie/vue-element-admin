#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from unittest.mock import MagicMock
from unittest.mock import patch
from op_test_frame.ut import OpUT

# pylint: disable=invalid-name
ut_case = OpUT("PRelu", "impl.dynamic.prelu", "prelu")

def side_effects(*args):
    return True

def test_v220_mock():
     with patch("te.platform.api_check_support",MagicMock(side_effect=side_effects)):
        from impl.dynamic.prelu import op_select_format as op_select_format_dynamic
        from impl.prelu import op_select_format as op_select_format_static
        op_select_format_dynamic({"shape": (32,32), "format": "NCHW", "dtype": "float16", "ori_shape": (32,32), "ori_format": "NCHW"},
                                 {"shape": (32,), "format": "NCHW", "dtype": "float16", "ori_shape": (32,), "ori_format": "NCHW"},
                                 {"shape": (32,32), "format": "NCHW", "dtype": "float16", "ori_shape": (32,32), "ori_format": "NCHW"})
        op_select_format_static({"shape": (32,32), "format": "NCHW", "dtype": "float16", "ori_shape": (32,32), "ori_format": "NCHW"},
                                {"shape": (32,), "format": "NCHW", "dtype": "float16", "ori_shape": (32,), "ori_format": "NCHW"},
                                {"shape": (32,32), "format": "NCHW", "dtype": "float16", "ori_shape": (32,32), "ori_format": "NCHW"})
case1 = {"params": [
    {"shape": (-2,), "dtype": "float32", "format": "NHWC", "ori_shape": (-2,),
     "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4)]},
    {"shape": (-2,), "dtype": "float32", "format": "NHWC", "ori_shape": (-2,),
     "ori_format": "NHWC", "range": [(1, 1), (1, 10), (1, 10)]},
    {"shape": (-2,), "dtype": "float32", "format": "NHWC", "ori_shape": (-2,),
     "ori_format": "NHWC", "range": [(1, 100), (2, 2), (4, 4), (4, 4)]}],
          "case_name": "prelu_dynamic_1",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    test_v220_mock()
    ut_case.run("Ascend910A")