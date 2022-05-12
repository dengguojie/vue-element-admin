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

ut_case.add_case(["Ascend910A"], case1)


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
