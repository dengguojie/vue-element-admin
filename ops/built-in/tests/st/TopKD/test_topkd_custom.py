#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from unittest.mock import MagicMock
from unittest.mock import patch
from impl.top_k_d import top_k_d



def side_effects(*args):
    return True

def test_v220_mock():
    with patch("te.platform.api_check_support",MagicMock(side_effect=side_effects)):
        with patch("te.tik.Tik.vmrgsort",MagicMock(side_effect=side_effects)):
            with patch("te.tik.Tik.vadds",MagicMock(side_effect=side_effects)):
                with patch("te.tik.Tik.vsort32",MagicMock(side_effect=side_effects)):
                    with patch("te.tik.Tik.vreduce",MagicMock(side_effect=side_effects)):
                        top_k_d({"shape": (100000,), "format": "ND", "dtype": "float16", "ori_shape": (100000,), "ori_format": "ND"},
                                {"shape": (8192, ), "format": "ND", "dtype": "float16", "ori_shape": (8192,), "ori_format": "ND"},
                                {"shape": (100000,), "format": "ND", "dtype": "float16", "ori_shape": (100000,), "ori_format": "ND"},
                                {"shape": (100000,), "format": "ND", "dtype": "int32", "ori_shape": (100000,), "ori_format": "ND"},
                                100000, True, -1, True)


if __name__ == '__main__':
    print("test v220 mock topk10w")
    test_v220_mock()
