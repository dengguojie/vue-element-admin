# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-

# from op_test_frame.ut import OpUT
# ut_case = OpUT("AvgPool", None, None)

# case1 = {"params": [{"shape": (1,2,32,32,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,32,32,16),"ori_format": "NC1HWC0"},
#                     {"shape": (2048,1,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2048,1,16,16),"ori_format": "NC1HWC0"},
#                     None,
#                     {"shape": (1,2,32,32,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,32,32,16),"ori_format": "NC1HWC0"},
#                     [1,32,32,1], [1,1,1,1], "VALID"],
#          "case_name": "avg_pool_1",
#          "expect": "success",
#          "format_expect": [],
#          "support_expect": True}



# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)


# if __name__ == '__main__':
#     ut_case.run("Ascend910")
#     exit(0)
