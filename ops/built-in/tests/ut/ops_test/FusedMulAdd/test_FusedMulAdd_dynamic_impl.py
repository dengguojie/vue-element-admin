#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import json


ut_case = OpUT("FusedMulAdd", "impl.dynamic.fused_mul_add", "fused_mul_add")


case1 = {
    "params": [
        {
            "shape": (-1, ),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(1, 3), ]
        },
        {
            "shape": (2, 2),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(2, 2), (2, 2)]
        },
        {
            "shape": (-1, ),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(1, 3), ]
        },
        {
            "shape": (-1, ),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(1, 3), ]
        }
    ],
    "case_name": "FusedMulAdd_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {
            "shape": (2, ),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(1, 3), ]
        },
        {
            "shape": (2, 2),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(2, 2), (2, 2)]
        },
        {
            "shape": (2, ),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(1, 3), ]
        },
        {
            "shape": (2, ),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2, ),
            "ori_format": "ND",
            "range": [(1, 3), ]
        }
    ],
    "case_name": "FusedMulAdd_2",
    "expect": "success",
    "op_imply_type": "static",
    "support_expect": True
}

case3 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_3",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}


case4 = {"params": [{"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_1",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}


case5 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_2",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_3",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_4",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [{"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_11",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}


case9 = {"params": [{"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_12",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}

case10 = {"params": [{"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_13",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}

case11 = {"params": [{"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_14",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}

case12 = {"params": [{"shape": (16, 1), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16, 1), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_22",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}

case13 = {"params": [{"shape": (16, 1), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (16, 1), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_23",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}

case14 = {"params": [{"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16, 1), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_24",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}


case15 = {"params": [{"shape": (1, 16), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1, 16), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_32",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}

case16 = {"params": [{"shape": (1, 16), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1, 16), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_33",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}

case17 = {"params": [{"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1, 16), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_pattern_34",
         "expect": "success",
         "op_imply_type": "static",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)
ut_case.add_case(["Ascend910A"], case9)
ut_case.add_case(["Ascend910A"], case10)
ut_case.add_case(["Ascend910A"], case11)
ut_case.add_case(["Ascend910A"], case12)
ut_case.add_case(["Ascend910A"], case13)
ut_case.add_case(["Ascend910A"], case14)
ut_case.add_case(["Ascend910A"], case15)
ut_case.add_case(["Ascend910A"], case16)
ut_case.add_case(["Ascend910A"], case17)


def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.dynamic.fused_mul_add import op_select_format
    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     "test_fused_mul_add_op_select_format_1")


    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     "test_fused_mul_add_op_select_format_2")
    
    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     "test_fused_mul_add_op_select_format_3")

    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"},
                     {"shape": (1, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 4), "ori_format": "ND"},
                     {"shape": (1, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 4), "ori_format": "ND"},
                     "test_fused_mul_add_op_select_format_4")
    
    op_select_format({"shape": (4, 16, 24, 24, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16, 24, 24, 16, 16), "ori_format": "ND"},
                     {"shape": [], "dtype": "float16", "format": "ND", "ori_shape": [], "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 16, 24, 24, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16, 24, 24, 16, 16), "ori_format": "ND"},
                     "test_fused_mul_add_op_select_format_5")

def test_op_support_info(test_arg):
    """
    test_op_support_info
    """
    from impl.dynamic.fused_mul_add import get_op_support_info
    res = get_op_support_info({"shape": (4, 16, 24, 24, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 16, 384, 384), "ori_format": "ND"},
                     {"shape": [], "dtype": "float16", "format": "ND", "ori_shape": [], "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 16, 24, 24, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16, 384, 384), "ori_format": "ND"},
                     "test_fused_mul_add_op_support_info_1")
    
    res_2 = get_op_support_info({"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     "test_fused_mul_add_op_support_info_2")
    
    split_maps = json.loads(res).get("_op_slice_info").get("splitMaps")
    assert len(split_maps) == 1
    for item in split_maps:
        input_list = item.get("inputList")
        assert len(input_list) == 2
        idx = input_list[0].get("idx")
        assert idx == 0
        idx = input_list[1].get("idx")
        assert idx == 2
    

ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_op_support_info)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
