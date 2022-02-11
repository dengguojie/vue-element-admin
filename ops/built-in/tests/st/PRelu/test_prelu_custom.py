#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from unittest.mock import MagicMock
from unittest.mock import patch


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


if __name__ == '__main__':
    test_v220_mock()