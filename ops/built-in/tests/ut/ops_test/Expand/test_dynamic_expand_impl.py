#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("Expand", "impl.dynamic.expand", "expand")


def gen_dynamic_broadcast_to_case(shape_x, range_x, shape, dtype_val,
                            kernel_name_val, expect):
    return {"params": [
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x},
        {"ori_shape": (len(shape),), "shape": (len(shape),), "ori_format": "ND",
         "format": "ND", "dtype": "int32", "range": ((1, 10),)},
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x}],
        "case_name": kernel_name_val, "expect": expect, "format_expect": [],
        "support_expect": True}


ut_case.add_case("Ascend910A", gen_dynamic_broadcast_to_case((1, 1, -1), ((1, 1),(1, 1), (5, 5)), [3, 1, 5],
                                                "float16",
                                                "dynamic_expand_1",
                                                "success"))

ut_case.add_case("Ascend910A", gen_dynamic_broadcast_to_case((3, 1, -1), ((3, 3),(1, 1), (5, 5)), [3, 1, 5],
                                                "float32",
                                                "dynamic_expand_2",
                                                "success"))

ut_case.add_case("Ascend910A", gen_dynamic_broadcast_to_case((1, 3, -1), ((1, 1),(3, 3), (5, 5)), [3, 1, 5],
                                                "float32",
                                                "dynamic_expand_3",
                                                "success"))

ut_case.add_case("Ascend910A", gen_dynamic_broadcast_to_case((3, 1, 5), (), [3, 2, 5],
                                                "float32",
                                                "dynamic_expand_4",
                                                "success"))

ut_case.add_case("Ascend910A", gen_dynamic_broadcast_to_case((1, 3, -1), ((1, 1),(3, 3), (0, 5)), [3, 1, 5],
                                                "float32",
                                                "dynamic_expand_5",
                                                "success"))

ut_case.add_case("Ascend910A", gen_dynamic_broadcast_to_case((1, 3, -1), ((1, 1),(3, 3), (0, 5)), [3, 1, 5],
                                                "int8",
                                                "dynamic_expand_6",
                                                "success"))

ut_case.add_case("Ascend910A", gen_dynamic_broadcast_to_case((1, 3, -1), ((1, 1),(3, 3), (0, 5)), [3, 1, 5],
                                                "int32",
                                                "dynamic_expand_7",
                                                "success"))

if __name__ == "__main__":
    ut_case.run("Ascend910A")
