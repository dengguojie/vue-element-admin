# TODO fix me, run failed
# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-
# from op_test_frame.ut import OpUT
#
# ut_case = OpUT("GNTrainingReduce", "impl.gn_training_reduce", "gn_training_reduce")
#
#
# # TODO add you test case
#
# def verify_gn_training_reduce(shape_x, shape_sum, shape_result,
#                               num_group, data_format,
#                               dtype, dtype_others,
#                               kernel_name, expect):
#     return {"params":
#                 [{"shape": shape_x, "dtype": dtype, "format": data_format},
#                  {"shape": shape_sum, "dtype": dtype_others, "format": data_format},
#                  {"shape": shape_sum, "dtype": dtype_others, "format": data_format},
#                  {"shape": shape_result, "dtype": dtype, "format": data_format},
#                  num_group],
#             "case_name": kernel_name,
#             "expect": expect}
#
# ut_case.add_case("all",
#                  verify_gn_training_reduce((64, 3, 224, 224), (64, 3, 1, 1, 1),
#                                            (64, 3, 1, 224, 224),
#                                            1, "NC2HWC0",
#                                            "float32", "float32",
#                                           "cce_gn_training_reduce",
#                                           RuntimeError))
#
# ut_case.add_case("all",
#                  verify_gn_training_reduce((64, 3, 224, 224), (64, 3, 1, 1, 1),
#                                            (64, 3, 1, 224, 224),
#                                            1, "NCHW",
#                                            "float64", "float32",
#                                            "cce_gn_training_reduce",
#                                            RuntimeError))
#
# ut_case.add_case("all",
#                  verify_gn_training_reduce((64, 3, 224, 224), (64, 3, 1, 1, 11),
#                                            (64, 3, 1, 224, 224),
#                                            2, "NCHW",
#                                            "float32", "float32",
#                                            "cce_gn_training_reduce",
#                                            RuntimeError))
#
# ut_case.add_case("all",
#                  verify_gn_training_reduce((64, 3, 224, 224, 1), (64, 3, 1, 1, 11),
#                                            (64, 3, 1, 224, 224),
#                                            2, "NCHW",
#                                            "float32", "float32",
#                                            "cce_gn_training_reduce",
#                                            RuntimeError))
#
# ut_case.add_case("all",
#                  verify_gn_training_reduce((64, 4, 224, 224), (64, 2, 1, 1, 1),
#                                            (64, 2, 2, 224, 224),
#                                            2, "NCHW",
#                                            "float32", "float32",
#                                            "cce_gn_training_reduce",
#                                            "success"))
#
# ut_case.add_case("all",
#                  verify_gn_training_reduce((64, 224, 224, 4), (64, 1, 1, 2, 1),
#                                            (64, 224, 224, 2, 2),
#                                            2, "NHWC",
#                                            "float32", "float32",
#                                            "cce_gn_training_reduce",
#                                            "success"))
#
# ut_case.add_case("all",
#                  verify_gn_training_reduce((64, 224, 224, 4), (64, 1, 1, 2, 1),
#                                            (64, 224, 224, 2, 2),
#                                            2, "NHWC",
#                                            "float16", "float32",
#                                            "cce_gn_training_reduce",
#                                            "success"))
#
# if __name__ == '__main__':
#     # ut_case.run("Ascend910")
#     ut_case.run()
#     exit(0)
