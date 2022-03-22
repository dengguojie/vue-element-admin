#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from impl.dynamic.one_hot import check_supported


def test_check_supported():
    input_x = {"shape": (240, 21128), "dtype": "int32", "format": "NCHW", "ori_shape": (240, 21128),
               "ori_format": "NCHW"}
    input_depth = {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"}
    input_on_val = {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"}
    input_off_val = {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"}
    axis = 1
    res, _ = check_supported(input_x, input_depth, input_on_val, input_off_val, input_x, axis)
    if not res:
        raise Exception("shape of input_x(240, 21128) is in white list, should return True")


if __name__ == '__main__':
    test_check_supported()
