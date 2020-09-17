# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-
# from op_test_frame.ut import OpUT
# ut_case = OpUT("MatMul", None, None)

# case1 = {"params": [{"shape": (96, 32), "dtype": "float16", "format": "ND", "ori_shape": (96, 32),"ori_format": "ND"},
#                     {"shape": (64, 96), "dtype": "float16", "format": "ND", "ori_shape": (64, 96),"ori_format": "ND"},
#                     {"shape": (32, 64), "dtype": "float16", "format": "ND", "ori_shape": (32, 64),"ori_format": "ND"},
#                     {},
#                     {"shape": (96, 32), "dtype": "float16", "format": "ND", "ori_shape": (96, 32),"ori_format": "ND"},
#                     True, True],
#          "case_name": "MatMul_1",
#          "expect": "success",
#          "format_expect": [],
#          "support_expect": True}



# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)

# if __name__ == '__main__':
#     ut_case.run("Ascend910")
