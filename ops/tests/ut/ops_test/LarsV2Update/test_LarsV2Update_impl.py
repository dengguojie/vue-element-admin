# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-

# from op_test_frame.ut import OpUT
# ut_case = OpUT("LarsV2Update", None, None)

# case1 = {"params": [{"shape": (1, 1, 512, 128), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1, 1, 512, 128)},
#                     {"shape": (1, 1, 512, 128), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1, 1, 512, 128)},
#                     {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
#                     {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
#                     {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
#                     {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
#                     {}],
#          "case_name": "lars_v2_update_1",
#          "expect": "success",
#          "format_expect": [],
#          "support_expect": True}
# case2 = {"params": [{"shape": (1, 1, 512, 128), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1, 1, 512, 128)},
#                     {"shape": (1, 1, 512, 128), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1, 1, 512, 128)},
#                     {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
#                     {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
#                     {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
#                     {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
#                     {}],
#          "case_name": "lars_v2_update_2",
#          "expect": "success",
#          "format_expect": [],
#          "support_expect": True}
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

# if __name__ == '__main__':
#     ut_case.run()
#     # ut_case.run("Ascend910")
#     exit(0)