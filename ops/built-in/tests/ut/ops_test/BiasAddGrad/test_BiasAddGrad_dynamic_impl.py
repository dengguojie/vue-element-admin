#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("BiasAddGrad", "impl.dynamic.bias_add_grad", "bias_add_grad")

case1 = {"params": [{"shape": (-1, -1, -1), "dtype": "float16", "format": "NHWC", "ori_shape": (2,3),"ori_format": "NHWC","range":[(1, 100),(1, 100),(1, 100)]}, #x
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NHWC", "ori_shape":(2,3),"ori_format": "NHWC","range":[(1, 100),(1, 100),(1, 100)]},
                    "NHWC"
                    ],
         "case_name": "BiasAddGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (2,3,4,5),"ori_format": "NHWC","range":[(1, 100),(1, 100),(1, 100), (1, 100)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape":(2,3,4,5),"ori_format": "NHWC","range":[(1, 100),(1, 100),(1, 100),(1, 100)]},
                    "NHWC"
                    ],
         "case_name": "BiasAddGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, 3, 4, 5), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (2,3,4,5),"ori_format": "NHWC","range":[(1, 100),(3, 3),(4, 4), (5, 5)]},
                    {"shape": (-1, 3, 4, 5), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape":(2,3,4,5),"ori_format": "NHWC","range":[(1, 100),(3, 3),(4, 4),(5, 5)]},
                    "NHWC"
                    ],
         "case_name": "BiasAddGrad_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (-1, 3, 4, -1), "dtype": "float16", "format": "FRACTAL_Z_3D", "ori_shape": (2,3,4,6),"ori_format": "NDHWC","range":[(1, 100),(3, 3),(4, 4),(1, 100)]},
                    {"shape": (-1, 3, 4, -1), "dtype": "float16", "format": "FRACTAL_Z_3D", "ori_shape":(2,3,4,6),"ori_format": "NDHWC","range":[(1, 100),(3, 3),(4, 4),(1, 100)]},
                    "NHWC"
                    ],
         "case_name": "BiasAddGrad_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (-1, 3, 4, 5), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (2,3,4,5),"ori_format": "NHWC","range":[(1, None),(3, 3),(4, 4), (5, 5)]},
                    {"shape": (-1, 3, 4, 5), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape":(2,3,4,5),"ori_format": "NHWC","range":[(1, 100),(3, 3),(4, 4),(5, 5)]},
                    "NHWC"
                    ],
         "case_name": "BiasAddGrad_5",
         "expect": "success",
         "support_expect": True}


ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case5)

if __name__ == '__main__':
    # ut_case.run(["Ascend310"])
    # ut_case.run(["Ascend710"])
    ut_case.run(["Ascend910A"])
