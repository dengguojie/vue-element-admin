#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("GNTrainingUpdate", "impl.gn_training_update", "gn_training_update")


# TODO add you test case

def verify_gn_training_update(shape_x, shape_sum, shape_square_sum,
                              shape_scale, shape_mean, shape_y,
                              data_format,
                              epsilon, num_groups,
                              dtype, dtype_others,
                              kernel_name, expect):
    if shape_mean == (0,):
        if shape_scale == (0,):
            return {"params":
                        [{"shape": shape_x, "dtype": dtype, "format": data_format},
                         {"shape": shape_sum, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_square_sum, "dtype": dtype_others, "format": data_format},
                         None, None, None, None,
                         {"shape": shape_y, "dtype": dtype, "format": data_format},
                         {"shape": shape_y, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_y, "dtype": dtype_others, "format": data_format},
                         epsilon, num_groups],
                    "case_name": kernel_name,
                    "expect": expect}
        else:
            return {"params":
                        [{"shape": shape_x, "dtype": dtype, "format": data_format},
                         {"shape": shape_sum, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_square_sum, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_scale, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_scale, "dtype": dtype_others, "format": data_format},
                         None, None,
                         {"shape": shape_y, "dtype": dtype, "format": data_format},
                         {"shape": shape_y, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_y, "dtype": dtype_others, "format": data_format},
                         epsilon, num_groups],
                    "case_name": kernel_name,
                    "expect": expect}
    else:
        if shape_scale == (0,):
            return {"params":
                        [{"shape": shape_x, "dtype": dtype, "format": data_format},
                         {"shape": shape_sum, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_square_sum, "dtype": dtype_others, "format": data_format},
                         None, None,
                         {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_y, "dtype": dtype, "format": data_format},
                         {"shape": shape_y, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_y, "dtype": dtype_others, "format": data_format},
                         epsilon, num_groups],
                    "case_name": kernel_name,
                    "expect": expect}
        else:
            return {"params":
                        [{"shape": shape_x, "dtype": dtype, "format": data_format},
                         {"shape": shape_sum, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_square_sum, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_scale, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_scale, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_mean, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_y, "dtype": dtype, "format": data_format},
                         {"shape": shape_y, "dtype": dtype_others, "format": data_format},
                         {"shape": shape_y, "dtype": dtype_others, "format": data_format},
                         epsilon, num_groups],
                    "case_name": kernel_name,
                    "expect": expect}


ut_case.add_case("all",
                 verify_gn_training_update((6, 5, 8, 7, 6), (6, 5, 1, 1, 3),
                                           (6, 5, 1, 1, 3), (6, 5, 1, 1, 3),
                                           (6, 5, 1, 1, 3), (6, 5, 1, 1, 3),
                                           "NC1HWC0", 0.0001, 2,
                                           "float32",
                                           "float32", "cce_group_norm_date_formate_error",
                                           RuntimeError))

ut_case.add_case("all",
                 verify_gn_training_update((6, 5, 8, 7), (6, 2, 3, 1, 1),
                                           (6, 2, 3, 1, 1), (1, 2, 3, 1, 1),
                                           (6, 2, 3, 1, 1), (6, 2, 3, 1, 1),
                                           "NCHW", 0.0001, 2,
                                           "float32",
                                           "float32", "cce_group_norm_channel_error",
                                           RuntimeError))

ut_case.add_case("all",
                 verify_gn_training_update((6, 1, 1), (6, 2, 3, 1, 1),
                                           (6, 2, 3, 1, 1), (1, 2, 3, 1, 1),
                                           (6, 2, 3, 1, 1), (6, 2, 3, 1, 1),
                                           "NCHW", 0.0001, 2,
                                           "float32",
                                           "float32", "cce_group_norm_dim_error",
                                           RuntimeError))

ut_case.add_case("all",
                 verify_gn_training_update((6, 6, 8, 7), (6, 2, 3, 1),
                                           (6, 2, 3, 1, 1), (1, 2, 3, 1, 1),
                                           (6, 2, 3, 1, 1), (6, 2, 3, 1, 1),
                                           "NCHW", 0.0001, 2,
                                           "float32",
                                           "float32", "cce_group_norm_dim_diff_error",
                                           RuntimeError))

ut_case.add_case("all",
                 verify_gn_training_update((6, 6, 1, 1), (6, 2, 3, 4, 1),
                                           (6, 2, 3, 1, 1), (1, 2, 3, 1, 1),
                                           (6, 2, 3, 1, 1), (6, 2, 3, 1, 1),
                                           "NCHW", 0.0001, 2,
                                           "float32",
                                           "float32", "cce_group_norm_dim_diff_error",
                                           RuntimeError))

ut_case.add_case("all",
                 verify_gn_training_update((6, 8, 7, 6), (6, 1, 43, 2, 3),
                                           (6, 1, 1, 2, 3), (1, 1, 1, 2, 3),
                                           (6, 1, 1, 2, 3), (6, 1, 1, 2, 3),
                                           "NHWC", 0.0001, 2,
                                           "float32",
                                           "float32", "cce_group_norm_dim_diff_error",
                                           RuntimeError))

ut_case.add_case("all",
                 verify_gn_training_update((6, 8, 7, 6), (6, 1, 1, 2, 3),
                                           (6, 1, 3, 2, 3), (1, 1, 1, 2, 3),
                                           (6, 1, 1, 2, 3), (6, 1, 1, 2, 3),
                                           "NHWC", 0.0001, 2,
                                           "float32",
                                           "float32", "cce_group_norm_dim_diff_error",
                                           RuntimeError))
# TODO fix me run failed
# ut_case.add_case("all",
#                  verify_gn_training_update((6, 6, 8, 7), (6, 2, 3, 1, 1),
#                                            (6, 2, 3, 1, 1), (1, 2, 3, 1, 1),
#                                            (6, 2, 3, 1, 1), (6, 2, 3, 1, 1),
#                                            "NCHW", 0.0001, 2,
#                                            "float32",
#                                            "float32", "cce_group_norm_dtype_error",
#                                            "success"))
#
# ut_case.add_case("all",
#                  verify_gn_training_update((6, 6, 8, 7), (6, 2, 3, 1, 1),
#                                            (6, 2, 3, 1, 1), (1, 2, 3, 1, 1),
#                                            (6, 2, 3, 1, 1), (6, 2, 3, 1, 1),
#                                            "NCHW", 0.0001, 2,
#                                            "float16",
#                                            "float32", "cce_group_norm_dtype_error",
#                                            "success"))
#
# ut_case.add_case("all",
#                  verify_gn_training_update((6, 8, 7, 6), (6, 1, 1, 2, 3),
#                                            (6, 1, 1, 2, 3), (1, 1, 1, 2, 3),
#                                            (6, 1, 1, 2, 3), (6, 1, 1, 2, 3),
#                                            "NHWC", 0.0001, 2,
#                                            "float32",
#                                            "float32", "cce_group_norm_dtype_error",
#                                            "success"))
#
#
# ut_case.add_case("all",
#                  verify_gn_training_update((6, 8, 7, 6), (6, 1, 1, 2, 3),
#                                            (6, 1, 1, 2, 3), (0,),
#                                            (6, 1, 1, 2, 3), (6, 1, 1, 2, 3),
#                                            "NHWC", 0.0001, 2,
#                                            "float32",
#                                            "float32", "cce_group_norm_dtype_error",
#                                            "success"))
#
# ut_case.add_case("all",
#                  verify_gn_training_update((6, 8, 7, 6), (6, 1, 1, 2, 3),
#                                            (6, 1, 1, 2, 3), (1, 1, 1, 2, 3),
#                                            (0,), (6, 1, 1, 2, 3),
#                                            "NHWC", 0.0001, 2,
#                                            "float32",
#                                            "float32", "cce_group_norm_dtype_error",
#                                            "success"))
#
# ut_case.add_case("all",
#                  verify_gn_training_update((6, 8, 7, 6), (6, 1, 1, 2, 3),
#                                            (6, 1, 1, 2, 3), (0,),
#                                            (0,), (6, 1, 1, 2, 3),
#                                            "NHWC", 0.0001, 2,
#                                            "float32",
#                                            "float32", "cce_group_norm_dtype_error",
#                                            "success"))

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
