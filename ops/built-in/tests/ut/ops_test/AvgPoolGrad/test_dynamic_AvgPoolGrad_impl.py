#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AvgPoolGrad", "impl.dynamic.avg_pool_grad", "avg_pool_grad")

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                     "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                    {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,1,3,3),"ori_format": "NCHW",
                     "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                    {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                     "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                    [1,1,3,3], [1,1,2,2], "SAME", "NCHW"],
         "expect": "success",
         "support_expect": True}
         
case2 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,16),"ori_format": "NHWC",
                     "range":[(1, 1), (1, 1), (1, 1), (16, 16)]},
                    {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,3,3,1),"ori_format": "NHWC",
                     "range":[(16, 16), (3, 3), (3, 3), (1, 1)]},
                    {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,16),"ori_format": "NHWC",
                     "range":[(1, 1), (4, 4), (4, 4), (16, 16)]},
                    [1,3,3,1], [1,2,2,1], "VALID", "NHWC"],
         "expect": "success",
         "support_expect": True}

#filter_h/w != ksize_h/w  
case3 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case4 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case5 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case6 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case7 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case8 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case9 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case10 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case11 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case12 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                     "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
                    {"shape": (-1,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,1,3,3),"ori_format": "NCHW",
                     "range":[(16, 16), (1, 1), (3, 3), (3, 3)]},
                    {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),"ori_format": "NCHW",
                     "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
                    [1,1,3,3], [1,1,2,2], "SAME", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)
ut_case.add_case(["Ascend910A"], case9)
ut_case.add_case(["Ascend910A"], case10)
ut_case.add_case(["Ascend910A"], case11)
ut_case.add_case(["Ascend910A"], case12)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
