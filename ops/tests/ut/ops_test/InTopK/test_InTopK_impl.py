# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-

# from op_test_frame.ut import OpUT
# ut_case = OpUT("InTopK", None, None)

# case1 = {"params": [{"shape": (1, 1), "dtype": "float32"},
#                     {"shape": (1, ), "dtype": "int32"},
#                     {},
#                     1],
#          "case_name": "in_top_k_1",
#          "expect": "success",
#          "format_expect": [],
#          "support_expect": True}

# case2 = {"params": [{"shape": (3, 231), "dtype": "float32"},
#                     {"shape": (3, ), "dtype": "int32"},
#                     {},
#                     1],
#          "case_name": "in_top_k_2",
#          "expect": "success",
#          "format_expect": [],
#          "support_expect": True}
# case3 = {"params": [{"shape": (3, 256), "dtype": "float32"},
#                     {"shape": (3, ), "dtype": "int32"},
#                     {},
#                     1],
#          "case_name": "in_top_k_3",
#          "expect": "success",
#          "format_expect": [],
#          "support_expect": True}
# case4 = {"params": [{"shape": (32, 63), "dtype": "float32"},
#                     {"shape": (32, ), "dtype": "int32"},
#                     {},
#                     1],
#          "case_name": "in_top_k_4",
#          "expect": "success",
#          "format_expect": [],
#          "support_expect": True}
# case5 = {"params": [{"shape": (13, 138), "dtype": "float32"},
#                     {"shape": (13, ), "dtype": "int32"},
#                     {},
#                     1],
#          "case_name": "in_top_k_5",
#          "expect": "success",
#          "format_expect": [],
#          "support_expect": True}

# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

# if __name__ == '__main__':
#     ut_case.run()
#     exit(0)