#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from unittest.mock import MagicMock
from unittest.mock import patch
from te import platform as cce_conf
from impl.util.platform_adapter import tbe_context
from impl.max_pool_grad import max_pool_grad as max_pool_grad_static


def test_max_pool_grad_static():
    input_list_01 = [{"shape": (18, 21, 224, 18, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (18, 224, 18, 336), "ori_format": "NHWC"},
                     {"shape": (18, 21, 5, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (18, 5, 3, 336), "ori_format": "NHWC"},
                     {"shape": (18, 21, 5, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (18, 5, 3, 336), "ori_format": "NHWC"},
                     {"shape": (18, 21, 224, 18, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (18, 224, 18, 336), "ori_format": "NHWC"},
                     [1, 7, 7, 1], [1, 48, 6, 1], "SAME"]
    max_pool_grad_static(*input_list_01)

    input_list_02 = [{"shape": (98, 5, 212, 17, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (98, 212, 17, 80), "ori_format": "NHWC"},
                     {"shape": (98, 5, 4, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (98, 4, 1, 80), "ori_format": "NHWC"},
                     {"shape": (98, 5, 4, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (98, 4, 1, 80), "ori_format": "NHWC"},
                     {"shape": (98, 5, 212, 17, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (98, 212, 17, 80), "ori_format": "NHWC"},
                     [1, 40, 1, 1], [1, 45, 55, 1], "VALID"]
    max_pool_grad_static(*input_list_02)

    input_list_03 = [{"shape": (81, 1, 122, 191, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (81, 122, 191, 16), "ori_format": "NHWC"},
                     {"shape": (81, 1, 61, 4, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (81, 61, 4, 16), "ori_format": "NHWC"},
                     {"shape": (81, 1, 61, 4, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (81, 61, 4, 16), "ori_format": "NHWC"},
                     {"shape": (81, 1, 122, 191, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (81, 122, 191, 16), "ori_format": "NHWC"},
                     [1, 6, 45, 1], [1, 2, 59, 1], "SAME"]
    max_pool_grad_static(*input_list_03)

    input_list_04 = [{"shape": (8, 46, 4, 285, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (8, 4, 285, 736), "ori_format": "NHWC"},
                     {"shape": (8, 46, 1, 7, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (8, 1, 7, 736), "ori_format": "NHWC"},
                     {"shape": (8, 46, 1, 7, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (8, 1, 7, 736), "ori_format": "NHWC"},
                     {"shape": (8, 46, 4, 285, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (8, 4, 285, 736), "ori_format": "NHWC"},
                     [1, 2, 87, 1], [1, 13, 30, 1], "VALID"]
    max_pool_grad_static(*input_list_04)

    input_list_05 = [{"shape": (1, 73, 356, 118, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 356, 118, 1168), "ori_format": "NHWC"},
                     {"shape": (1, 73, 71, 2, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 71, 2, 1168), "ori_format": "NHWC"},
                     {"shape": (1, 73, 71, 2, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 71, 2, 1168), "ori_format": "NHWC"},
                     {"shape": (1, 73, 356, 118, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 356, 118, 1168), "ori_format": "NHWC"},
                     [1, 2, 24, 1], [1, 5, 53, 1], "VALID"]
    max_pool_grad_static(*input_list_05)

    input_list_06 = [{"shape": (88, 2, 111, 182, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (88, 111, 182, 32), "ori_format": "NHWC"},
                     {"shape": (88, 2, 9, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (88, 9, 3, 32), "ori_format": "NHWC"},
                     {"shape": (88, 2, 9, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (88, 9, 3, 32), "ori_format": "NHWC"},
                     {"shape": (88, 2, 111, 182, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (88, 111, 182, 32), "ori_format": "NHWC"},
                     [1, 2, 129, 1], [1, 13, 21, 1], "VALID"]
    max_pool_grad_static(*input_list_06)

    input_list_07 = [{"shape": (7, 1, 204, 565, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (7, 204, 565, 16), "ori_format": "NHWC"},
                     {"shape": (7, 1, 29, 160, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (7, 29, 160, 16), "ori_format": "NHWC"},
                     {"shape": (7, 1, 29, 160, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (7, 29, 160, 16), "ori_format": "NHWC"},
                     {"shape": (7, 1, 204, 565, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (7, 204, 565, 16), "ori_format": "NHWC"},
                     [1, 4, 88, 1], [1, 7, 3, 1], "VALID"]
    max_pool_grad_static(*input_list_07)

    input_list_08 = [{"shape": (42, 2, 101, 303, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (42, 101, 303, 32), "ori_format": "NHWC"},
                     {"shape": (42, 2, 17, 101, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (42, 17, 101, 32), "ori_format": "NHWC"},
                     {"shape": (42, 2, 17, 101, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (42, 17, 101, 32), "ori_format": "NHWC"},
                     {"shape": (42, 2, 101, 303, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (42, 101, 303, 32), "ori_format": "NHWC"},
                     [1, 2, 81, 1], [1, 6, 3, 1], "SAME"]
    max_pool_grad_static(*input_list_08)

    input_list_09 = [{"shape": (9, 5, 586, 120, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (9, 586, 120, 80), "ori_format": "NHWC"},
                     {"shape": (9, 5, 31, 10, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (9, 31, 10, 80), "ori_format": "NHWC"},
                     {"shape": (9, 5, 31, 10, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (9, 31, 10, 80), "ori_format": "NHWC"},
                     {"shape": (9, 5, 586, 120, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (9, 586, 120, 80), "ori_format": "NHWC"},
                     [1, 44, 1, 1], [1, 19, 12, 1], "SAME"]
    max_pool_grad_static(*input_list_09)


if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Ascend910A")
    vals = {("tik.load3dv1", ): False}
    def side_effects(*args):
        return vals[args]
    with patch("te.platform.cce_conf.api_check_support", MagicMock(side_effect=side_effects)):
        test_max_pool_grad_static()

    cce_conf.te_set_version(soc_version)
