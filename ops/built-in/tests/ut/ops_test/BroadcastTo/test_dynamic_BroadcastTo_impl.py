#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("BroadcastTo", "impl.dynamic.broadcast_to", "broadcast_to")


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
                                                "dynamic_broadcast_to_fp16_ND",
                                                "success"))

dynamic_rank_case_1 = {
    "params": [
        {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
        {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
        {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"}
    ],
    "case_name": "case_1",
    "expect": "success",
    "support_expect": True
}

dynamic_rank_case_2 = {
    "params": [
        {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
        {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,), "ori_format": "ND", "range": ((1, 100),)},
        {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"}
    ],
    "case_name": "case_2",
    "expect": "success",
    "support_expect": True
}

dynamic_rank_case_3 = {
    "params": [
        {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,), "ori_format": "ND", "range": ((1, 100),)},
        {"shape": (-2,), "dtype": "int32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"},
        {"shape": (-2,), "dtype": "float32", "format": "ND", "ori_shape": (-2,), "ori_format": "ND"}
    ],
    "case_name": "case_3",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend310"], dynamic_rank_case_1)
ut_case.add_case(["Ascend910A", "Ascend310"], dynamic_rank_case_2)
ut_case.add_case(["Ascend910A", "Ascend310"], dynamic_rank_case_3)

if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend310"])
