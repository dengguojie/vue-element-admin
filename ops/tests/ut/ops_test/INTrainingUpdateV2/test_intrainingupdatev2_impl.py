# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-
# from op_test_frame.ut import OpUT
#
# ut_case = OpUT("INTrainingUpdateV2", "impl.in_training_update_v2", "in_training_update_v2")
#
#
# # TODO add you test case
# # TODO fix me, run failed, case right 1, 2, 4
# def verify_in_training_update_v2(shape_x, shape_sum, shape_sqrt_sum,
#                                  shape_weight, shape_mean,
#                                  data_format,
#                                  momentum, epsilon,
#                                  dtype, dtype_others,
#                                  kernel_name, expect):
#     if shape_mean == (0,):
#         if shape_weight == (0,):
#             return {"params":
#                         [{"shape": shape_x, "dtype": dtype, "format": data_format},
#                          {"shape": shape_sum, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_sqrt_sum, "dtype": dtype_others, "format": data_format},
#                          None, None, None, None,
#                          {"shape": shape_x, "dtype": dtype, "format": data_format},
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          momentum, epsilon],
#                     "case_name": kernel_name,
#                     "expect": expect}
#         else:
#             return {"params":
#                         [{"shape": shape_x, "dtype": dtype, "format": data_format},
#                          {"shape": shape_sum, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_sqrt_sum, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_weight, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_weight, "dtype": dtype_others, "format": data_format},
#                          None, None,
#                          {"shape": shape_x, "dtype": dtype, "format": data_format},
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          momentum, epsilon],
#                     "case_name": kernel_name,
#                     "expect": expect}
#     else:
#         if shape_weight == (0,):
#             return {"params":
#                         [{"shape": shape_x, "dtype": dtype, "format": data_format},
#                          {"shape": shape_sum, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_sqrt_sum, "dtype": dtype_others, "format": data_format},
#                          None, None,
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_x, "dtype": dtype, "format": data_format},
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          momentum, epsilon],
#                     "case_name": kernel_name,
#                     "expect": expect}
#         else:
#             return {"params":
#                         [{"shape": shape_x, "dtype": dtype, "format": data_format},
#                          {"shape": shape_sum, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_sqrt_sum, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_weight, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_weight, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_x, "dtype": dtype, "format": data_format},
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
#                          momentum, epsilon],
#                     "case_name": kernel_name,
#                     "expect": expect}
#
#
# ut_case.add_case("all",
#                  verify_in_training_update_v2((6, 5, 8, 7, 6), (6, 5, 1, 1, 6),
#                                               (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
#                                               (6, 5, 1, 1, 6),
#                                               "NC1HWC0", 0.001, 0.0001,
#                                               "float64",
#                                               "float32",
#                                               "test_error_001",
#                                               RuntimeError))
#
# ut_case.add_case("all",
#                  verify_in_training_update_v2((6, 5, 8, 7), (6, 5, 1, 1, 6),
#                                               (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
#                                               (6, 5, 1, 1, 6),
#                                               "NC1HWC0", 0.001, 0.0001,
#                                               "float32",
#                                               "float32",
#                                               "test_error_002",
#                                               RuntimeError))
#
# ut_case.add_case("all",
#                  verify_in_training_update_v2((6, 5, 8, 7, 6), (6, 5, 1, 1, 5),
#                                               (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
#                                               (6, 5, 1, 1, 6),
#                                               "NC1HWC0", 0.001, 0.0001,
#                                               "float32",
#                                               "float32",
#                                               "test_error_003",
#                                               RuntimeError))
#
# ut_case.add_case("all",
#                  verify_in_training_update_v2((6, 5, 8, 7, 6), (6, 5, 2, 1, 6),
#                                               (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
#                                               (6, 5, 1, 1, 6),
#                                               "NC1HWC0", 0.001, 0.0001,
#                                               "float32",
#                                               "float32",
#                                               "test_error_004",
#                                               RuntimeError))
#
# ut_case.add_case("all",
#                  verify_in_training_update_v2((6, 5, 8, 7, 6), (6, 5, 2, 1, 6),
#                                               (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
#                                               (6, 5, 1, 1, 6),
#                                               "NC2HWC0", 0.001, 0.0001,
#                                               "float32",
#                                               "float32",
#                                               "test_error_005",
#                                               RuntimeError))
#
# ut_case.add_case("all",
#                  verify_in_training_update_v2((6, 5, 8, 7, 6), (6, 5, 2, 1, 6),
#                                               (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
#                                               (6, 5, 1, 1, 6),
#                                               "NC1HWC0", 0.001, 0.0001,
#                                               "float64",
#                                               "float32",
#                                               "test_error_006",
#                                               RuntimeError))
#
# # ut_case.add_case("all",
# #                  verify_in_training_update_v2((6, 5, 8, 7, 6), (6, 5, 1, 1, 6),
# #                                               (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
# #                                               (6, 5, 1, 1, 6),
# #                                               "NC1HWC0", 0.001, 0.0001,
# #                                               "float32",
# #                                               "float32",
# #                                               "test_right_001",
# #                                               "success"))
# #
# # ut_case.add_case("all",
# #                  verify_in_training_update_v2((6, 5, 8, 7, 6), (6, 5, 1, 1, 6),
# #                                               (6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
# #                                               (0,),
# #                                               "NC1HWC0", 0.001, 0.0001,
# #                                               "float16",
# #                                               "float32",
# #                                               "test_right_002",
# #                                               "success"))
#
# ut_case.add_case("all",
#                  verify_in_training_update_v2((6, 5, 8, 7, 6), (6, 5, 1, 1, 6),
#                                               (6, 5, 1, 1, 6), (0,),
#                                               (6, 5, 1, 1, 6),
#                                               "NC1HWC0", 0.001, 0.0001,
#                                               "float16",
#                                               "float32",
#                                               "test_right_003",
#                                               "success"))
#
# # ut_case.add_case("all",
# #                  verify_in_training_update_v2((6, 5, 8, 7, 6), (6, 5, 1, 1, 6),
# #                                               (6, 5, 1, 1, 6), (0,),
# #                                               (0,),
# #                                               "NC1HWC0", 0.001, 0.0001,
# #                                               "float16",
# #                                               "float32",
# #                                               "test_right_004",
# #                                               "success"))
#
# ut_case.add_case("all",
#                  verify_in_training_update_v2((6, 5, 1, 1, 6), (6, 5, 1, 1, 6),
#                                               (6, 5, 1, 1, 6), (0,),
#                                               (0,),
#                                               "NC1HWC0", 0.001, 0.0001,
#                                               "float16",
#                                               "float32",
#                                               "test_right_005",
#                                               "success"))
#
# if __name__ == '__main__':
#     # ut_case.run("Ascend910")
#     ut_case.run()
#     exit(0)
