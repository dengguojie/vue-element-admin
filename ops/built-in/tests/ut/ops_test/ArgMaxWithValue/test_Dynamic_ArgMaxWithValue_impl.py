#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=invalid-name,missing-module-docstring,too-many-arguments
# pylint: disable=consider-using-sys-exit,import-error,missing-function-docstring,missing-docstring
from op_test_frame.ut import OpUT

ut_case = OpUT("ArgMaxWithValue", "impl.dynamic.arg_max_with_value", "arg_max_with_value")


def gen_dynamic_argmax_case(shape_x, shape_y, ori_shape_x, ori_shape_y, range_x, range_y, in_format, ori_format,
                            dtype_val, axis, kernel_name_val, expect):
    return {
        "params": [{
            "shape": shape_x,
            "dtype": dtype_val,
            "ori_shape": ori_shape_x,
            "ori_format": ori_format,
            "format": in_format,
            "range": range_x
        }, {
            "shape": shape_y,
            "dtype": dtype_val,
            "ori_shape": ori_shape_y,
            "ori_format": ori_format,
            "format": in_format,
            "range": range_y
        }, {
            "shape": shape_y,
            "dtype": dtype_val,
            "ori_shape": ori_shape_y,
            "ori_format": ori_format,
            "format": in_format,
            "range": range_y
        }, axis],
        "case_name": kernel_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True
    }


ut_case.add_case(
    "all",
    gen_dynamic_argmax_case((-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1),
                            ((1, None), (1, None), (1, None), (1, None), (1, None)),
                            ((1, None), (1, None), (1, None), (1, None), (1, None)), "ND", "ND", "float16",
                            1, "argmax_case", "success"))
ut_case.add_case(
    ["Ascend910A"],
    gen_dynamic_argmax_case((-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1),
                            ((1, None), (1, None), (1, None), (1, None), (1, None)),
                            ((1, None), (1, None), (1, None), (1, None), (1, None)), "ND", "ND", "float32",
                            1, "argmax_case", "success"))
ut_case.add_case(
    "all",
    gen_dynamic_argmax_case((-2,), (-2,), (-2,), (-2,), ((1, None),), ((1, None),), "ND", "ND", "float16",
                            1, "argmax_case", "success"))

if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)
