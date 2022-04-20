#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from unittest.mock import MagicMock
from unittest.mock import patch
from impl.dynamic.floor_mod import floor_mod as dynamic_floor_mod
from impl.floor_mod import floor_mod as static_floor_mod
from impl.floor_mod import check_supported as check_supported_static
from impl.dynamic.floor_mod import check_supported as check_supported_dynamic
import tbe


def side_effects(*args):
    return True

def test_dynamic_floor_mod():
    input_x1 = {"ori_shape": (16,), "shape": (16,), "ori_format": "ND",
                "format": "ND", "dtype": "int32", "range": ((16, 16),)}
    input_x2 = {"ori_shape": (16,), "shape": (16,), "ori_format": "ND",
                "format": "ND", "dtype": "int32", "range": ((16, 16),)}
    output_y = {"ori_shape": (16,), "shape": (16,), "ori_format": "ND",
                "format": "ND", "dtype": "int32", "range": ((16, 16),)}

    dynamic_floor_mod(input_x1, input_x2, output_y, impl_mode="high_precision")

def test_static_floor_mod():
    input_x1 = {"shape": (10, 12), "dtype": "int32", "format": "ND", "ori_shape": (10, 12),"ori_format": "ND"}
    input_x2 = {"shape": (10, 12), "dtype": "int32", "format": "ND", "ori_shape": (10, 12),"ori_format": "ND"}
    output_y = {"shape": (10, 12), "dtype": "int32", "format": "ND", "ori_shape": (10, 12),"ori_format": "ND"}

    static_floor_mod(input_x1, input_x2, output_y, impl_mode="high_precision")


def test_floor_mod_mock():
    with patch("te.platform.api_check_support",MagicMock(side_effect=side_effects)):
        test_static_floor_mod()
        # test_dynamic_floor_mod()


def test_check_supported_001(): 
    def side_effects(*args):
        context = tbe.common.context.op_context.OpContext()
        context.add_addition("op_impl_mode_dict", {"FloorMod": "high_precision"})
        return context

    with patch('tbe.common.context.op_context.get_context', MagicMock(side_effect=side_effects)):
        check_supported_static({"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        "floor_mod",
                        "high_precision")

        check_supported_dynamic({"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        "floor_mod",
                        "high_precision")

def test_check_supported_002(): 
    def side_effects(*args):
        context = tbe.common.context.op_context.OpContext()
        context.add_addition("op_impl_mode_dict", {"FloorMod": "high_preformance"})
        return context

    with patch('tbe.common.context.op_context.get_context', MagicMock(side_effect=side_effects)):
        check_supported_static({"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        {"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        {"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        "floor_mod",
                        "high_precision")

        check_supported_dynamic({"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        {"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        {"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        "floor_mod",
                        "high_precision")

def test_floor_mod_check_supported():
    test_check_supported_001()
    test_check_supported_002()

if __name__ == '__main__':
    test_floor_mod_mock()
    test_floor_mod_check_supported()
