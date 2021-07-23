#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AvgPoolGrad", "impl.dynamic.avg_pool_grad", "avg_pool_grad")

# dynamic_nw SAME NCHW range None 
case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case2 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case18 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case15 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case13 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case14 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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

#-2 VALID
case16 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
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
case17 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1, -1, -1, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 1, -1),"ori_format": "NHWC",
                     "range":[(1, 1), (2, 99), (2, 99), (17, 17)]}, 
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (17, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(17, 17), (1, 1), (2, 2), (2, 2)]},
                    {"shape": (1,-1,-1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, 2, -1), "ori_format": "NHWC",
                     "range":[(1, 1), (2, 100), (2, 100), (17, 17)]},
                    [1,2,2,1], [1,1,2,1], "SAME", "NHWC"],
         "expect": "success",
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
ut_case.add_case(["Ascend910A"], case13)
ut_case.add_case(["Ascend910A"], case14)
ut_case.add_case(["Ascend910A"], case15)
ut_case.add_case(["Ascend910A"], case16)
ut_case.add_case(["Ascend910A"], case17)
ut_case.add_case(["Ascend910A"], case18)
ut_case.add_cust_test_func(test_func=test_avg_pool_grad_fuzzy_compile_generaliation)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
