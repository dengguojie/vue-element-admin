#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AvgPoolGrad", "impl.dynamic.avg_pool_grad", "avg_pool_grad")

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
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_generaliation)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
