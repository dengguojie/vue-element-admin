#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from impl.dynamic.avg_pool_grad import avg_pool_grad_generalization
from impl.dynamic.avg_pool_grad import UNSUPPORTED_FUZZ_RES

ut_case = OpUT("AvgPoolGrad", "impl.dynamic.avg_pool_grad", "avg_pool_grad")

LOWER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [1], "type": ["lower_limit"]}}]
UPPER_STR = [{"result": "UNSUPPORTED", "reason": {"param_index": [1], "type": ["upper_limit"]}}]

# dynamic_nw SAME NCHW range None
dynamic_nw_SAME_NCHW_range_None = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                              {"shape": (-1,1,2,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,16,1,-1),"ori_format": "NCHW",
                                               "range":[(1, None), (16, 16), (1, 1), (1, None)]},
                                              {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,1,3,3),"ori_format": "NCHW",
                                               "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                                              {"shape": (-1,1,1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,16,1,-1),"ori_format": "NCHW",
                                               "range":[(1, None), (16, 16), (1, 1), (1, None)]},
                                               [1,1,3,3], [1,1,2,2], "SAME", "NCHW"],
                                   "expect": "success",
                                   "support_expect": True}

#dynamic_nhw VALID NHWC
dynamic_nhw_VALID_NHWC = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                     {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,-1,-1,16),"ori_format": "NHWC",
                                      "range":[(1, 1), (3, 10), (3, 10), (16, 16)]},
                                     {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,3,3,1),"ori_format": "NHWC",
                                      "range":[(16, 16), (3, 3), (3, 3), (1, 1)]},
                                     {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,-1,-1,16),"ori_format": "NHWC",
                                      "range":[(1, 10), (3, 4), (3, 4), (16, 16)]},
                                      [1,3,3,1], [1,2,2,1], "VALID", "NHWC"],
                          "expect": "success",
                          "support_expect": True}

#dx hw dynamic dy not
dx_hw_dynamic_dy_n = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                 {"shape": (-1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,3,3,16),"ori_format": "NHWC",
                                  "range":[(1, 1), (3, 10), (3, 10), (16, 16)]},
                                 {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,3,3,1),"ori_format": "NHWC",
                                  "range":[(16, 16), (3, 3), (3, 3), (1, 1)]},
                                 {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,-1,-1,16),"ori_format": "NHWC",
                                  "range":[(1, 10), (3, 4), (3, 4), (16, 16)]},
                                  [1,3,3,1], [1,2,2,1], "VALID", "NHWC"],
                      "expect": "success",
                      "support_expect": True}

# dynamic hw SAME NHWC
dynamic_hw_SAME_NHWC = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                    {"shape": (1, 2, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, -1, 32),"ori_format": "NHWC",
                                    "range":[(1, 1), (2, 99), (2, 99), (32, 32)]},
                                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                                    "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                                    {"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,32), "ori_format": "NHWC",
                                    "range":[(1, 1), (2, 100), (2, 100), (32, 32)]},
                                    [1,2,2,1], [1,1,2,1], "SAME", "NHWC"],
                        "expect": "success",
                        "support_expect": True}

# dynamic n SAME NCHW
dynamic_n_SAME_NCHW = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                    {"shape": (-1, 2, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, 32, 1, 1),"ori_format": "NCHW",
                                    "range":[(1, 10), (32, 32), (1, 99), (1, 99)]},
                                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                                    "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                                    {"shape": (-1,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,32,1,1), "ori_format": "NCHW",
                                    "range":[(1, 1), (32, 32), (2, 100), (2, 100)]},
                                    [1,1,2,2], [1,1,2,1], "SAME", "NCHW"],
                        "expect": "success",
                        "support_expect": True}

# dynamic h VALID NHWC
dynamic_h_VALID_NHWC = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                    {"shape": (1, 2, -1, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 1, 32),"ori_format": "NHWC",
                                    "range":[(1, 1), (2, 99), (2, 99), (32, 32)]},
                                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                                    "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                                    {"shape": (1,2,-1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 2, 32), "ori_format": "NHWC",
                                    "range":[(1, 1), (1, 100), (1, 100), (32, 32)]},
                                    [1,2,2,1], [1,2,1,1], "VALID", "NHWC"],
                        "expect": "success",
                        "support_expect": True}


#filter_h/w != ksize_h/w
filter_not_equal_ksize = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                        {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                        "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                                        {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,1,3,3),"ori_format": "NCHW",
                                        "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                                        {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                        "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                                        [1,1,30,30], [1,1,2,2], "SAME", "NCHW"],
                        "expect": RuntimeError,
                        "support_expect": True}

#stride_h/w < 1
stride_h_and_stride_w_less_than_zero = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                                    {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                                    "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                                                    {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,1,3,3),"ori_format": "NCHW",
                                                    "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                                                    {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                                    "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                                                    [1,1,3,3], [1,1,0,0], "SAME", "NCHW"],
                                        "expect": RuntimeError,
                                        "support_expect": True}

#stride_n/c != 1
stride_n_and_stride_c_not_equal_one = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                                    {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                                    "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                                                    {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,1,3,3),"ori_format": "NCHW",
                                                    "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                                                    {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                                    "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                                                    [1,1,3,3], [2,2,2,2], "SAME", "NCHW"],
                                        "expect": RuntimeError,
                                        "support_expect": True}

#dx_n != dy_n, dx_c != dy_c
dx_n_and_c_not_equal_dy = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                        {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,32,-1,-1),"ori_format": "NCHW",
                                        "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                                        {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,1,3,3),"ori_format": "NCHW",
                                        "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                                        {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                        "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                                        [1,1,3,3], [1,1,2,2], "SAME", "NCHW"],
                            "expect": RuntimeError,
                            "support_expect": True}

#dx_c != k_n
dx_c_not_equal_filter_n = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                        {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                        "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                                        {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,1,3,3),"ori_format": "NCHW",
                                        "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                                        {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                        "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                                        [1,1,3,3], [1,1,2,2], "SAME", "NCHW"],
                            "expect": RuntimeError,
                            "support_expect": True}

#k_c != 1
filter_c_not_equal_one = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                        {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                        "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                                        {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,2,3,3),"ori_format": "NCHW",
                                        "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                                        {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                        "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                                        [1,1,3,3], [1,1,2,2], "SAME", "NCHW"],
                            "expect": RuntimeError,
                            "support_expect": True}

#k_h/w > 255
filter_h_and_w_great_than_255 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                            {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                            "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                                            {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,1,256,256),"ori_format": "NCHW",
                                            "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                                            {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                            "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                                            [1,1,3,3], [1,1,2,2], "SAME", "NCHW"],
                                "expect": RuntimeError,
                                "support_expect": True}

#-2
dx_shape_is_neg_two = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                    {"shape": (-2,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-2,),"ori_format": "NCHW",
                                    "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                                    {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,1,3,3),"ori_format": "NCHW",
                                    "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                                    {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                    "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                                    [1,1,3,3], [1,1,2,2], "SAME", "NCHW"],
                        "expect": "success",
                        "support_expect": True}

#-1
dyanmic_nhw = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                            {"shape": (1,-1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,-1),"ori_format": "NCHW",
                            "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                            {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,1,3,3),"ori_format": "NCHW",
                            "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                            {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                            "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                            [1,1,3,3], [1,1,2,2], "SAME", "NCHW"],
                "expect": "success",
                "support_expect": True}

#cout = -1
filter_n_equal_neg_one = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                        {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                        "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                                        {"shape": (-1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,1,3,3),"ori_format": "NCHW",
                                        "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                                        {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                                        "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                                        [1,1,3,3], [1,1,2,2], "SAME", "NCHW"],
                            "expect": RuntimeError,
                            "support_expect": True}

#-2 VALID
dx_shape_neg_two_valid = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                        {"shape": (-2,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-2,),"ori_format": "NHWC",
                                        "range": None},
                                        {"shape": (7,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (7,1,3,3),"ori_format": "NCHW",
                                        "range":[(7, 7), (1, 1), (3, 3), (3, 3)]},
                                        {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,-1,-1,7),"ori_format": "NHWC",
                                        "range":[(1, -1), (1, -1), (1, -1), (7, 7)]},
                                        [1,3,3,1], [1,2,2,1], "SAME", "NHWC"],
                            "expect": "success",
                            "support_expect": True}

# dynamic hc SAME NHWC
dynamic_hc_same_NHWC = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                    {"shape": (1, -1, -1, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 1, -1),"ori_format": "NHWC",
                                    "range":[(1, 1), (2, 99), (2, 99), (17, 17)]},
                                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (17, 1, 2, 2),"ori_format": "NCHW",
                                    "range":[(17, 17), (1, 1), (2, 2), (2, 2)]},
                                    {"shape": (1,-1,-1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 2, -1), "ori_format": "NHWC",
                                    "range":[(1, 1), (2, 100), (2, 100), (17, 17)]},
                                    [1,2,2,1], [1,1,2,1], "SAME", "NHWC"],
                        "expect": "success",
                        "support_expect": True}

dx_h_range_dim_less_than_2 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                            {"shape": (1, -1, -1, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 1, -1),"ori_format": "NHWC",
                                            "range":[(1, 1), (2, 99), (2, 99), (17, 17)]},
                                            {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (17, 1, 2, 2),"ori_format": "NCHW",
                                            "range":[(17, 17), (1, 1), (2, 2), (2, 2)]},
                                            {"shape": (1,-1,-1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 2, -1), "ori_format": "NHWC",
                                            "range":[(1, 1), (2,), (2, 100), (17, 17)]},
                                            [1,2,2,1], [1,1,2,1], "SAME", "NHWC"],
                                "expect": RuntimeError,
                                "support_expect": True}

dx_h_range_lower_bound_less_than_1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                                    {"shape": (1, -1, -1, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 1, -1),"ori_format": "NHWC",
                                                    "range":[(1, 1), (0, 99), (2, 99), (17, 17)]},
                                                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (17, 1, 2, 2),"ori_format": "NCHW",
                                                    "range":[(17, 17), (1, 1), (2, 2), (2, 2)]},
                                                    {"shape": (1,-1,-1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 2, -1), "ori_format": "NHWC",
                                                    "range":[(1, 1), (2, 100), (2, 100), (17, 17)]},
                                                    [1,2,2,1], [1,1,2,1], "SAME", "NHWC"],
                                        "expect": RuntimeError,
                                        "support_expect": True}

dx_h_range_lower_bound_grest_than_upper_bound = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                                            {"shape": (1, -1, -1, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 1, -1),"ori_format": "NHWC",
                                                            "range":[(1, 1), (99, 2), (2, 99), (17, 17)]},
                                                            {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (17, 1, 2, 2),"ori_format": "NCHW",
                                                            "range":[(17, 17), (1, 1), (2, 2), (2, 2)]},
                                                            {"shape": (1,-1,-1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 2, -1), "ori_format": "NHWC",
                                                            "range":[(1, 1), (2, 100), (2, 100), (17, 17)]},
                                                            [1,2,2,1], [1,1,2,1], "SAME", "NHWC"],
                                                "expect": RuntimeError,
                                                "support_expect": True}

data_format_ND = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                            {"shape": (1, -1, -1, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 1, -1),"ori_format": "NHWC",
                            "range":[(1, 1), (99, 2), (2, 99), (17, 17)]},
                            {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (17, 1, 2, 2),"ori_format": "NCHW",
                            "range":[(17, 17), (1, 1), (2, 2), (2, 2)]},
                            {"shape": (1,-1,-1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 2, -1), "ori_format": "NHWC",
                            "range":[(1, 1), (2, 100), (2, 100), (17, 17)]},
                            [1,2,2,1], [1,1,2,1], "SAME", "ND"],
                "expect": RuntimeError,
                "support_expect": True}

filter_ori_format_ND = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                                    {"shape": (1, -1, -1, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 1, -1),"ori_format": "NHWC",
                                    "range":[(1, 1), (99, 2), (2, 99), (17, 17)]},
                                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (17, 1, 2, 2),"ori_format": "ND",
                                    "range":[(17, 17), (1, 1), (2, 2), (2, 2)]},
                                    {"shape": (1,-1,-1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 2, -1), "ori_format": "NHWC",
                                    "range":[(1, 1), (2, 100), (2, 100), (17, 17)]},
                                    [1,2,2,1], [1,1,2,1], "SAME", "NHWC"],
                            "expect": RuntimeError,
                            "support_expect": True}

def test_avg_pool_grad_fuzzy_compile_generaliation(test_arg):
    from impl.dynamic.avg_pool_grad import avg_pool_grad_generalization
    print("test_avg_pool_grad_fuzzy_compile_generaliation")
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'format': 'NCHW',
            'ori_format': 'NCHW',
            'dtype': 'float16'
        }, {
            'shape': (1, 1, 2, 17, 16),
            'ori_shape': (1, 5, 2, 17),
            'format': 'NC1HWC0',
            'ori_format': 'NCHW',
            'dtype': 'float16',
            'range': ((1, 1), (1, 1), (1, 3), (16, 31), (16, 16)),
            'ori_range': ((1, 1), (5, 5), (1, 3), (16, 31))
        }, {
            'shape': (49, 1, 16, 16),
            'ori_shape': (7, 7, 1, 5),
            'format': 'FRACTAL_Z',
            'ori_format': 'HWCN',
            'dtype': 'float16'
        }, {
            'shape': (1, 1, 4, 66, 16),
            'ori_shape': (1, 5, 4, 66),
            'format': 'NC1HWC0',
            'ori_format': 'NCHW',
            'dtype': 'float16'
        },
        (1, 1, 7, 7),
        (1, 1, 3, 4),
        'SAME',
        'NCHW',
        'avg_pool_grad_fuzzy_compile_generaliation',
        {'mode': 'keep_rank'}
    ]
    avg_pool_grad_generalization(*input_list)

def test_avg_pool_grad_fuzzy_compile_dedy_w_too_large(test_arg):
    from impl.dynamic.avg_pool_grad import avg_pool_grad_generalization
    print("test_avg_pool_grad_fuzzy_compile_dedy_w_too_large")
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'format': 'NCHW',
            'ori_format': 'NCHW',
            'dtype': 'float16'
        }, {
            'shape': (3, 1, 281, 590, 16),
            'ori_shape': (3, 1, 281, 590),
            'format': 'NC1HWC0',
            'ori_format': 'NCHW',
            'dtype': 'float16',
            'range': ((2, 3), (1, 1), (256, 511), (512, 767), (16, 16)),
            'ori_range': ((2, 3), (1, 1), (256, 511), (512, 767))
        }, {
            'shape': (288, 1, 16, 16),
            'ori_shape': (18, 16, 1, 1),
            'format': 'FRACTAL_Z',
            'ori_format': 'HWCN',
            'dtype': 'float16'
        }, {
            'shape': (3, 1, 841, 1768, 16),
            'ori_shape': (3, 1, 841, 1768),
            'format': 'NC1HWC0',
            'ori_format': 'NCHW',
            'dtype': 'float16'
        },
        (1, 1, 18, 16),
        (1, 1, 3, 3),
        'SAME',
        'NCHW',
        'avg_pool_grad_fuzzy_compile_generaliation',
        {'mode': 'keep_rank'}
    ]

    try:
        avg_pool_grad_generalization(*input_list)
    except Exception as e:
        print(e)

def test_avg_pool_grad_fuzzy_compile_strideHW_1_dedy_w_too_large(test_arg):
    from impl.dynamic.avg_pool_grad import avg_pool_grad_generalization
    print("test_avg_pool_grad_fuzzy_compile_dedy_w_too_large")
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'format': 'NCHW',
            'ori_format': 'NCHW',
            'dtype': 'float16'
        }, {
            'shape': (1, 1, 1, 1024, 16),
            'ori_shape': (1, 1, 1, 1024),
            'format': 'NC1HWC0',
            'ori_format': 'NCHW',
            'dtype': 'float16',
            'range': ((1, 1), (1, 1), (1, 3), (1024, 4096), (16, 16)),
            'ori_range': ((1, 1), (1, 1), (1, 3), (1024, 4096))
        }, {
            'shape': (16, 1, 16, 16),
            'ori_shape': (16, 1, 1, 1),
            'format': 'FRACTAL_Z',
            'ori_format': 'HWCN',
            'dtype': 'float16'
        }, {
            'shape': (1, 1, 1, 1024, 16),
            'ori_shape': (1, 1, 1, 1024),
            'format': 'NC1HWC0',
            'ori_format': 'NCHW',
            'dtype': 'float16'
        },
        (16, 1, 16, 16),
        (1, 1, 2, 1),
        'SAME',
        'NCHW',
        'avg_pool_grad_fuzzy_compile_generaliation',
        {'mode': 'keep_rank'}
    ]

    try:
        avg_pool_grad_generalization(*input_list)
    except Exception as e:
        print(e)


def test_avg_pool_grad_fuzzy_compile_head_node_1(test_arg):
    # input_size, dedy, filter, dedx, ksize, strides, padding
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 26, 28), 'shape': (1, 1, 26, 28, 16) ,'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (16, 16), (26, 26), (28, 28)),
         'range': ((1, 1), (1, 1), (26, 26), (28, 28), (16, 16))},
        {'ori_shape': (16, 1, 3, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 28, 30), 'shape': (1, 1, 28, 30, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
         'range': ((1, 1), (1, 1), (28, 28), (30, 30), (16, 16))},
        (16, 1, 3, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_1',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == [[
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16', 'const_value': None,
        'const_value_range': [(1, 1), (1, 1), (18, 33), (18, 33)]},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 26, 28, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (16, 16), [16, 31], [16, 31]], 'range': ((1, 1), (1, 1), (26, 26), (28, 28), (16, 16))},
        {'ori_shape': (16, 1, 3, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 28, 30, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'range': ((1, 1), (1, 1), (28, 28), (30, 30), (16, 16))},
        {'strides': (1, 1, 1, 1)}, {'padding': 'VALID'}, {'ksize': (16, 1, 3, 3)},
        {'kernel_name': 'test_avg_pool_grad_fuzzy_compile_head_node_1'}, {'data_format': 'NCHW'}]]


def test_avg_pool_grad_fuzzy_compile_head_node_2(test_arg):
    # N dim exceeds
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (21474836548, 32, 26, 28), 'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (32, 32), (26, 26), (28, 28))},
        {'ori_shape': (32, 16, 3, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 28, 30), 'ori_format': 'NCHW', 'dtype': 'float16'},
        (32, 16, 3, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_2',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_avg_pool_grad_fuzzy_compile_head_node_3(test_arg):
    # H/W dim exceeds limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 32, 4900, 28), 'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (32, 32), (26, 26), (28, 28))},
        {'ori_shape': (32, 16, 3, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 28, 30), 'ori_format': 'NCHW', 'dtype': 'float16'},
        (32, 16, 3, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_3',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_avg_pool_grad_fuzzy_compile_head_node_4(test_arg):
    # [-2]
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (-2,), 'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': None},
        {'ori_shape': (32, 16, 3, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 28, 30), 'ori_format': 'NCHW', 'dtype': 'float16'},
        (32, 16, 3, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_4',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_avg_pool_grad_fuzzy_compile_head_node_5(test_arg):
    # dedy exceeds l1 size limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 32, 60, 1000), 'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (32, 32), (60, 60), (1000, 1000))},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 90, 1002), 'ori_format': 'NCHW', 'dtype': 'float16'},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_5',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UNSUPPORTED_FUZZ_RES


def test_avg_pool_grad_fuzzy_compile_head_node_6(test_arg):
    # dedy exceeds l1 size limit, correct w range
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 60, 900), 'shape': (1, 1, 60, 900, 16) ,'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (16, 16), (60, 60), (900, 900)),
         'range': ((1, 1), (1, 1), (60, 60), (900, 900), (16, 16))},
        {'ori_shape': (16, 1, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 90, 902), 'shape': (1, 1, 90, 902, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
         'range': ((1, 1), (1, 1), (90, 90), (902, 902), (16, 16))},
        (16, 1, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_6',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == [[
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16', 'const_value': None,
        'const_value_range': [(1, 1), (1, 1), (62, 93), (770, 979)]},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 60, 900, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (16, 16), [32, 63], [768, 977]],
        'range': ((1, 1), (1, 1), (60, 60), (900, 900), (16, 16))},
        {'ori_shape': (16, 1, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 90, 902, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'range': ((1, 1), (1, 1), (90, 90), (902, 902), (16, 16))},
        {'strides': (1, 1, 1, 1)}, {'padding': 'VALID'}, {'ksize': (16, 1, 31, 3)},
        {'kernel_name': 'test_avg_pool_grad_fuzzy_compile_head_node_6'}, {'data_format': 'NCHW'}]]


def test_avg_pool_grad_fuzzy_compile_head_node_7(test_arg):
    # kernel_matrix is None
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': (1, 16, 26, 28), 'shape': (1, 1, 26, 28, 16) ,'ori_format': 'NCHW', 'dtype': 'float16',
         'ori_range': ((1, 1), (16, 16), (26, 26), (28, 28)),
         'range': ((1, 1), (1, 1), (26, 26), (28, 28), (16, 16))},
        None,
        {'ori_shape': (1, 16, 28, 30), 'shape': (1, 1, 28, 30, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
         'range': ((1, 1), (1, 1), (28, 28), (30, 30), (16, 16))},
        (16, 1, 3, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_head_node_7',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == [[
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16', 'const_value': None,
        'const_value_range': [(1, 1), (1, 1), (18, 33), (18, 33)]},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 26, 28, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (16, 16), [16, 31], [16, 31]], 'range': ((1, 1), (1, 1), (26, 26), (28, 28), (16, 16))},
        {'ori_shape': [16, 1, 3, 3], 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (1, 1, 28, 30, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'range': ((1, 1), (1, 1), (28, 28), (30, 30), (16, 16))},
        {'strides': (1, 1, 1, 1)}, {'padding': 'VALID'}, {'ksize': (16, 1, 3, 3)},
        {'kernel_name': 'test_avg_pool_grad_fuzzy_compile_head_node_7'}, {'data_format': 'NCHW'}]]


def test_avg_pool_grad_fuzzy_compile_body_node_1(test_arg):
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape':(-1, 1, -1, -1, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 63], [768, 977]],
        'range': [[1, 1], (1, 1), [32, 63], [768, 977], [16, 16]]},
        {'ori_shape': (16, 1, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape':(-1, 1, -1, -1, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)],
        'range': [(1, 1), (1, 1), (62, 93), (770, 979), (16, 16)]},
        (16, 1, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_1',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == [[
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (-1, 1, -1, -1, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 63], [768, 977]],
        'range': [[1, 1], (1, 1), [32, 63], [768, 977], [16, 16]]},
        {'ori_shape': (16, 1, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'shape': (-1, 1, -1, -1, 16), 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)],
        'range': [(1, 1), (1, 1), (62, 93), (770, 979), (16, 16)]}, {'strides': (1, 1, 1, 1)},
        {'padding': 'VALID'}, {'ksize': (16, 1, 31, 3)},
        {'kernel_name': 'test_avg_pool_grad_fuzzy_compile_body_node_1'}, {'data_format': 'NCHW'}]]


def test_avg_pool_grad_fuzzy_compile_body_node_2(test_arg):
    # N dim lower bound exceeds limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 32, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[21474836548, 1], (32, 32), [32, 63], [768, 977]]},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)]},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_2',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == LOWER_STR


def test_avg_pool_grad_fuzzy_compile_body_node_3(test_arg):
    # H dim upper bound exceeds limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 32, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 4100], [768, 977]]},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)]},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_3',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UPPER_STR


def test_avg_pool_grad_fuzzy_compile_body_node_4(test_arg):
    # W dim lower bound exceeds limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 32, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 63], [4100, 977]]},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)]},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_4',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == LOWER_STR


def test_avg_pool_grad_fuzzy_compile_body_node_5(test_arg):
    # W dim upper bound is -1
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 32, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 63], [768, -1]]},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 979)]},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_5',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UPPER_STR


def test_avg_pool_grad_fuzzy_compile_body_node_6(test_arg):
    # dedy exceeds l1 size limit
    input_list = [
        {'ori_shape': (4,), 'format': 'NCHW', 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 32, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [[1, 1], (32, 32), [32, 63], [768, 1000]]},
        {'ori_shape': (32, 16, 31, 3), 'ori_format': 'NCHW', 'dtype': 'float16'},
        {'ori_shape': [-1, 16, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16',
        'ori_range': [(1, 1), (16, 16), (62, 93), (770, 1000)]},
        (32, 16, 31, 3),
        (1, 1, 1, 1),
        'VALID',
        'NCHW',
        'test_avg_pool_grad_fuzzy_compile_body_node_6',
        {'mode': 'keep_rank'}
    ]
    res = avg_pool_grad_generalization(*input_list)
    assert res == UPPER_STR


ut_case.add_case(["Ascend910A"], dynamic_nw_SAME_NCHW_range_None)
ut_case.add_case(["Ascend910A"], dynamic_nhw_VALID_NHWC)
ut_case.add_case(["Ascend910A"], dx_hw_dynamic_dy_n)
ut_case.add_case(["Ascend910A"], dynamic_hw_SAME_NHWC)
ut_case.add_case(["Ascend910A"], dynamic_n_SAME_NCHW)
ut_case.add_case(["Ascend910A"], dynamic_h_VALID_NHWC)
ut_case.add_case(["Ascend910A"], filter_not_equal_ksize)
ut_case.add_case(["Ascend910A"], stride_h_and_stride_w_less_than_zero)
ut_case.add_case(["Ascend910A"], stride_n_and_stride_c_not_equal_one)
ut_case.add_case(["Ascend910A"], dx_n_and_c_not_equal_dy)
ut_case.add_case(["Ascend910A"], dx_c_not_equal_filter_n)
ut_case.add_case(["Ascend910A"], filter_c_not_equal_one)
ut_case.add_case(["Ascend910A"], filter_h_and_w_great_than_255)
ut_case.add_case(["Ascend910A"], dx_shape_is_neg_two)
ut_case.add_case(["Ascend910A"], dyanmic_nhw)
ut_case.add_case(["Ascend910A"], filter_n_equal_neg_one)
ut_case.add_case(["Ascend910A"], dx_shape_neg_two_valid)
ut_case.add_case(["Ascend910A"], dynamic_hc_same_NHWC)
ut_case.add_case(["Ascend910A"], dx_h_range_dim_less_than_2)
ut_case.add_case(["Ascend910A"], dx_h_range_lower_bound_less_than_1)
ut_case.add_case(["Ascend910A"], dx_h_range_lower_bound_grest_than_upper_bound)
ut_case.add_case(["Ascend910A"], data_format_ND)
ut_case.add_case(["Ascend910A"], filter_ori_format_ND)
# ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_generaliation)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_dedy_w_too_large)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_strideHW_1_dedy_w_too_large)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_head_node_1)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_head_node_2)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_head_node_3)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_head_node_4)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_head_node_5)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_head_node_6)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_head_node_7)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_body_node_1)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_body_node_2)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_body_node_3)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_body_node_4)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_body_node_5)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_body_node_6)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
