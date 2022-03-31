#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from impl.inplace_update_d import check_supported


def test_check_supported():
    res, _ = check_supported({"shape": (30001, 3), "dtype": "float16", "format": "ND", "ori_shape": (30001, 3), "ori_format": "ND"},
                    {"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (1, 3), "ori_format": "ND"},
                    {"shape": (30001, 3), "dtype": "float16", "format": "ND", "ori_shape": (30001, 3), "ori_format": "ND"},
                    [-1],"inplace_update_d_check_support_case_001")

if __name__ == '__main__':
    test_check_supported()
