#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=invalid-name,missing-module-docstring,too-many-arguments
from op_test_frame.ut import OpUT

ut_case = OpUT("ScatterNDUpdate", "impl.dynamic.scatter_nd_update", "scatter_nd_update")


def gen_dynamic_scatter_case(shape_x, shape_y, range_x, range_y, dtype_val, kernel_name_val, expect):
    return {
        "params": [{
            "shape": shape_x,
            "dtype": dtype_val,
            "ori_shape": shape_x,
            "ori_format": "ND",
            "format": "ND",
            "range": range_x
        }, {
            "shape": shape_y,
            "dtype": "int32",
            "ori_shape": shape_y,
            "ori_format": "ND",
            "format": "ND",
            "range": range_y
        }, {
            "shape": shape_x,
            "dtype": dtype_val,
            "ori_shape": shape_x,
            "ori_format": "ND",
            "format": "ND",
            "range": range_x
        }, {
            "shape": shape_y,
            "dtype": dtype_val,
            "ori_shape": shape_y,
            "ori_format": "ND",
            "format": "ND",
            "range": range_y
        }],
        "case_name": kernel_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True
    }


ut_case.add_case(
    "all",
    gen_dynamic_scatter_case((-1,), (1,), ((1, None),), ((1, 1),), "float32", "dynamic_scatter_update_case_1",
                             "success"))

ut_case.add_case(
    "all",
    gen_dynamic_scatter_case((-2,), (-2,), ((1, None),), ((1, None),), "float16", "dynamic_scatter_update_case_2",
                             "success"))

def gen_dynamic_scatter_case_bool(shape_x, shape_y, range_x, range_y, dtype_val, kernel_name_val, expect):
    return {
        "params": [{
            "shape": shape_x,
            "dtype": 'int8',
            "ori_shape": shape_x,
            "ori_format": "ND",
            "format": "ND",
            "range": range_x
        }, {
            "shape": shape_y,
            "dtype": "int32",
            "ori_shape": shape_y,
            "ori_format": "ND",
            "format": "ND",
            "range": range_y
        }, {
            "shape": shape_x,
            "dtype": 'int8',
            "ori_shape": shape_x,
            "ori_format": "ND",
            "format": "ND",
            "range": range_x
        }, {
            "shape": shape_y,
            "dtype": 'bool',
            "ori_shape": shape_y,
            "ori_format": "ND",
            "format": "ND",
            "range": range_y
        }],
        "case_name": kernel_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True
    }

ut_case.add_case(
    "all",
    gen_dynamic_scatter_case_bool((-2,), (-2,), ((1, None),), ((1, None),), "float16", "dynamic_scatter_update_case_2",
                             "success"))


if __name__ == '__main__':
    ut_case.run("Ascend910A")
