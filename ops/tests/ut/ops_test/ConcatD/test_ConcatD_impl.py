# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-

# from op_test_frame.ut import OpUT
# ut_case = OpUT("ConcatD", None, None)

# case1 = {"params": [[{"shape": (4, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (4, 3, 3),"ori_format": "NCHW"},
#                      {"shape": (4, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (4, 3, 3),"ori_format": "NCHW"}],
#                     {"shape": (4, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (4, 3, 3),"ori_format": "NCHW"},
#                     1],
#          "case_name": "concat_d_1",
#          "expect": "success",
#          "format_expect": [],
#          "support_expect": True}

# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)

# if __name__ == '__main__':
#     ut_case.run("Ascend910")