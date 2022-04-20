#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
ut_test_dynamic_apply_add_sign_d
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("SquareSumV1",
               "impl.dynamic.square_sum_v1",
               "square_sum_v1")

case1 = {"params": [{"shape": (1, -1, -1), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 2, 4), "ori_format": "ND", "range": [(1, 1), (1, None), (1, None)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND",
                     "ori_shape": (4,), "ori_format": "ND", "range": [(1, None)]},
                    [0, 1], None],
         "case_name": "dynamic_square_sum_v1_001",
         "expect": "success"}

case2 = {"params": [{"shape": (-1, -1, -1), "dtype": "float32", "format": "ND",
                     "ori_shape": (16, 16, 16), "ori_format": "ND", "range": [(1, 16), (1, None), (1, None)]},
                    {"shape": (16,), "dtype": "float32", "format": "ND",
                     "ori_shape": (16,), "ori_format": "ND", "range": [(1, 16)]},
                    [1, 2], True],
         "case_name": "dynamic_square_sum_v1_002",
         "expect": "success"}

case3 = {"params": [{"shape": (3, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (16, 16), "ori_format": "ND"},
                    {"shape": (16, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (16, 16), "ori_format": "ND"},
                    [0, 1]],
         "case_name": "static_square_sum_v1_001",
         "expect": "success"}

ut_case.add_case(["Ascend310", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend910A"], case3)


def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.dynamic.square_sum_v1 import op_select_format
    op_select_format({"shape": (1, 1, 16, 16), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 16),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 16, 16), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 16),
                      "ori_format": "HWCN"},
                     [0,1,2,3],
                     attr2=True,
                     kernel_name="test_square_sum_v1_op_select_format_1")
    op_select_format({"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16),
                      "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16),
                      "ori_format": "ND"},
                     None,
                     attr2=True,
                     kernel_name="test_square_sum_v1_op_select_format_2")


def test_get_op_support_info(test_arg):
    """
    test_get_op_support_info
    """
    from impl.dynamic.square_sum_v1 import get_op_support_info
    get_op_support_info({"shape": (1, 1, 16, 16), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 16),
                         "ori_format": "HWCN"},
                        {"shape": (1, 1, 16, 16), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 16),
                         "ori_format": "HWCN"},
                        [0, 1, 2, 3],
                        attr2=True,
                        kernel_name="test_square_sum_v1_op_select_format_1")
    get_op_support_info({"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16),
                         "ori_format": "ND"},
                        {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16),
                         "ori_format": "ND"},
                        None,
                        attr2=True,
                        kernel_name="test_square_sum_v1_op_select_format_2")


ut_case.add_cust_test_func(test_func=test_get_op_support_info)
ut_case.add_cust_test_func(test_func=test_op_select_format)


if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend310"])

