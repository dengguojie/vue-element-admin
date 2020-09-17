# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-

# from op_test_frame.ut import OpUT
# ut_case = OpUT("MaxPoolGradGrad", None, None)


# ori_x_input = {"shape": (1, 1, 5, 5, 16),
#                "dtype": "float16",
#                "ori_shape": (1, 5, 5, 16),
#                "format": "5HD",
#                "ori_format": "NHWC"}
# ori_y_input = {"shape": (1, 1, 5, 5, 16),
#                "dtype": "float16",
#                "ori_shape": (1, 5, 5, 16),
#                "format": "NC1HWC0",
#                "ori_format": "NHWC"}
# grads = ori_x_input
# output = ori_y_input
# case1 = {"params":[ori_x_input, ori_y_input, grads, output, (1,3,3,1), (1,1,1,1)],
#          "case_name": "max_pool_grad_grad_1",
#          "expect": "success",
#          "format_expect": [],
#          "support_expect": True}

# ori_x_input = {"shape": (1, 1, 112, 224, 16),
#                "dtype": "float16",
#                "ori_shape": (1, 112, 224, 16),
#                "format": "5HD",
#                "ori_format": "NHWC"}
# ori_y_input = {"shape": (1, 1, 56, 112, 16),
#                "dtype": "float16",
#                "ori_shape": (1, 112, 224, 16),
#                "format": "NC1HWC0",
#                "ori_format": "NHWC"}
# grads = ori_x_input
# output = ori_y_input
# case2 = {"params":[ori_x_input, ori_y_input, grads, output, (1,1,3,3), (1,1,2,2)],
#          "case_name": "max_pool_grad_grad_2",
#          "expect": RuntimeError,
#          "format_expect": [],
#          "support_expect": True}



# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

# if __name__ == '__main__':
#     ut_case.run()
#     # ut_case.run("Ascend910")
#     exit(0)