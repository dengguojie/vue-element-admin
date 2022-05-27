#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("CdistGrad", "impl.dynamic.cdist_grad", "cdist_grad")


def gen_dynamic_cdist_case(shape_grad, range_grad, shape_x1, range_x1, shape_x2, range_x2, shape_cdist,
    range_cdist, shape_y, range_y, dtype_val, data_format, ori_shape_grad, ori_shape_x1, ori_shape_x2, ori_shape_cdist,
    ori_shape_y, p, kernel_name_val, expect, run_type):

    return {
        "params": [{
            "shape": shape_grad,
            "dtype": dtype_val,
            "range": range_grad,
            "format": data_format,
            "ori_shape": ori_shape_grad,
            "ori_format": data_format
        }, {
            "shape": shape_x1,
            "dtype": dtype_val,
            "range": range_x1,
            "format": data_format,
            "ori_shape": ori_shape_x1,
            "ori_format": data_format
        }, {
            "shape": shape_x2,
            "dtype": dtype_val,
            "range": range_x2,
            "format": data_format,
            "ori_shape": ori_shape_x2,
            "ori_format": data_format
        }, {
            "shape": shape_cdist,
            "dtype": dtype_val,
            "range": range_cdist,
            "format": data_format,
            "ori_shape": ori_shape_cdist,
            "ori_format": data_format
        }, {
            "shape": shape_y,
            "dtype": dtype_val,
            "range": range_y,
            "format": data_format,
            "ori_shape": ori_shape_y,
            "ori_format": data_format
        }, p],
        "case_name": kernel_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True,
        "op_impl_type": run_type
    }

ut_case.add_case(
    "all",
    gen_dynamic_cdist_case(
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 8), [(1, None), (8, 8)],
    "float16", "ND",
    (5, 4, 8), (5, 4, 8), (5, 4, 8), (5, 4, 8),
    (5, 8), 2.0, "dynamic_cdist_grad_fp16_ND",
    "success", "dynamic"))

ut_case.add_case(
    "all",
    gen_dynamic_cdist_case(
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 8), [(1, None), (8, 8)],
    "float16", "ND",
    (5, 4, 8), (5, 4, 8), (5, 4, 8), (5, 4, 8),
    (5, 8), 1.0, "dynamic_cdist_grad_fp16_ND",
    "success", "dynamic"))

ut_case.add_case(
    "all",
    gen_dynamic_cdist_case(
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 4, 8), [(1, None), (4, 4), (8, 8)],
    (-1, 8), [(1, None), (8, 8)],
    "float16", "ND",
    (5, 4, 8), (5, 4, 8), (5, 4, 8), (5, 4, 8),
    (5, 8), 0.0, "dynamic_cdist_grad_fp16_ND",
    "success", "dynamic"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
