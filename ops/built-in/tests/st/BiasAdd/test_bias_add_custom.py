#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from te import platform as cce_conf
from impl.bias_add import op_select_format

def test_op_select_format():
    op_select_format({"shape": (10, 10, 10, 16), "dtype": "float16", "format": "ND", "ori_shape": (10, 10, 10, 16), "ori_format": "ND"},
                     {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16, ), "ori_format": "ND"},
                     {"shape": (10, 10, 10, 16), "dtype": "float16", "format": "ND", "ori_shape": (10, 10, 10, 16), "ori_format": "ND"},
                     "test_add_op_select_format_1")
    op_select_format({"shape": (1, 1, 1, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 10),
                      "ori_format": "NHWC"},
                     {"shape": (10, ), "dtype": "float16", "format": "NHWC", "ori_shape": (10, ),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 1, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 10),
                      "ori_format": "NHWC"},
                     "test_add_op_select_format_2")
    op_select_format({"shape": (1, 1, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 10),
                      "ori_format": "NHWC"},
                     {"shape": (10, ), "dtype": "float16", "format": "NHWC", "ori_shape": (10, ),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 10),
                      "ori_format": "NHWC"},
                     "test_add_op_select_format_3")
    op_select_format({"shape": (1, 1, 1, 10, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 10, 16),
                      "ori_format": "NHWC"},
                     {"shape": (16, ), "dtype": "float16", "format": "NHWC", "ori_shape": (16, ),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 1, 10, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 10, 16),
                      "ori_format": "NHWC"},
                     "test_add_op_select_format_4")
    op_select_format({"shape": (1, 1, 1, 10, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 10, 16),
                      "ori_format": "NHWC"},
                     {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 1, 10, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 10, 16),
                      "ori_format": "NHWC"},
                     "test_add_op_select_format_5")

if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Hi3796CV300CS")
    test_op_select_format()
    cce_conf.te_set_version(soc_version)
