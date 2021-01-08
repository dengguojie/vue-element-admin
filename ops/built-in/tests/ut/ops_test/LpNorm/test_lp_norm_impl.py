# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("lp_norm", None, None)

case1 = {"params": [{"shape": (400, 416, 5, 69), "dtype": "float32", "format": "ND", "ori_shape": (400, 416, 5, 69), "ori_format": "ND"},
                    {"shape": (400, 416, 5, 1), "dtype": "float32", "format": "ND", "ori_shape": (400, 416, 5, 1), "ori_format": "ND"},
                    2147483647, [-1], True],
         "case_name": "lp_norm_1",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case2 = {"params": [{"shape": (400, 416, 5, 69), "dtype": "float32", "format": "ND", "ori_shape": (400, 416, 5, 69), "ori_format": "ND"},
                    {"shape": (400, 416, 5), "dtype": "float32", "format": "ND", "ori_shape": (400, 416, 5), "ori_format": "ND"},
                    -2147483648, [-1]],
         "case_name": "lp_norm_2",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float32", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"},
                    2, [0, 1]],
         "case_name": "lp_norm_3",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float32", "format": "ND", "ori_shape": (4, 16), "ori_format": "ND"},
                    "-inf", [0, 1]],
         "case_name": "lp_norm_4",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    "inf", [0, 1]],
         "case_name": "lp_norm_5",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case6 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float32", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"},
                    0, [0, 1]],
         "case_name": "lp_norm_6",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case7 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float32", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"},
                    1, [0, 1]],
         "case_name": "lp_norm_7",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
case8 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16), "ori_format": "ND"},
                    {"shape": (4, 16), "dtype": "float32", "format": "ND", "ori_shape": (4, 16),"ori_format": "ND"},
                    3, [0, 1]],
         "case_name": "lp_norm_8",
         "expect": "success",
         "format_expect": ["ND"],
         "support_expect": True}
ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)
ut_case.add_case("Ascend910A", case4)
ut_case.add_case("Ascend910A", case5)
ut_case.add_case("Ascend910A", case6)
ut_case.add_case("Ascend910A", case7)
ut_case.add_case("Ascend910A", case8)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
