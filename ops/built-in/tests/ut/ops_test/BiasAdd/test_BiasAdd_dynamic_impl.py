#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te

from op_test_frame.ut import OpUT
ut_case = OpUT("BiasAdd", "impl.dynamic.bias_add", "bias_add")

case1 = {"params": [{"shape": (-1,-1,4), "dtype": "float16", "format": "NHWC", "ori_shape": (-1,-1,4),"ori_format": "NHWC","range":[(1, 100),(1, 100),(4, 4)]}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NHWC", "ori_shape": (4,),"ori_format": "NHWC","range":[(4, 4)]},
                    {"shape": (-1,-1,4), "dtype": "float16", "format": "NHWC", "ori_shape":(-1,-1,4),"ori_format": "NHWC","range":[(1, 100),(1, 100),(4, 4)]},
                    ],
         "case_name": "BiasAdd_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,-1,-1,16),"ori_format": "NHWC","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,),"ori_format": "NHWC","range":[(4, 4)]},
                    {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,-1,-1,16),"ori_format": "NHWC","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}
                    ],
         "case_name": "BiasAdd_2",
         "expect": RuntimeError,
         "support_expect": True}

case3 = {"params": [{"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,-1,-1,16),"ori_format": "NHWC","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NHWC", "ori_shape": (4,),"ori_format": "NHWC","range":[(4, 4)]},
                    {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,-1,-1,16),"ori_format": "NHWC","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}
                    ],
         "case_name": "BiasAdd_3",
         "expect": RuntimeError,
         "support_expect": True}

case4 = {"params": [{"shape": (-1,16,1,-1,-1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(16, 16),(1, 1),(1, 100),(1, 100),(16, 16)]}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (4,),"ori_format": "NCDHW","range":[(4, 4)]},
                    {"shape": (-1,16,1,-1,-1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(16, 16),(1, 1),(1, 100),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (-1,16,1,-1,-1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(16, 16),(1, 1),(1, 100),(1, 100),(16, 16)]}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NDHWC", "ori_shape": (4,),"ori_format": "NCDHW","range":[(4, 4)]},
                    {"shape": (-1,16,1,-1,-1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(16, 16),(1, 1),(1, 100),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_5",
         "expect": RuntimeError,
         "support_expect": True}

case6 = {"params": [{"shape": (-1,1,-1,16), "dtype": "float16", "format": "NDHWC", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NDHWC","range":[(1, 100),(1, 1),(1, 100),(16, 16)]}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NDHWC", "ori_shape": (4,),"ori_format": "NCDHW","range":[(4, 4)]},
                    {"shape": (-1,1,-1,16), "dtype": "float16", "format": "NDHWC", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NDHWC","range":[(1, 100),(1, 1),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_6",
         "expect": RuntimeError,
         "support_expect": True}
case7 = {"params": [{"shape": (-1,1,-1,16), "dtype": "float16", "format": "NCDHW", "ori_shape": (-1,1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(16, 16)]}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NDHWC", "ori_shape": (4,),"ori_format": "NCDHW","range":[(4, 4)]},
                    {"shape": (-1,1,-1,16), "dtype": "float16", "format": "NDHWC", "ori_shape": (-1,1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_7",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": [{"shape": (-1,-1,-1, -1,-1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NDHWC","range":[(1, 100),(1, 1),(1, 100),(16, 16)]}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NDHWC", "ori_shape": (4,),"ori_format": "NCDHW","range":[(4, 4)]},
                    {"shape": (-1,-1,-1, -1,-1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NDHWC","range":[(1, 100),(1, 1),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_8",
         "expect": RuntimeError,
         "support_expect": True}
case9 = {"params": [{"shape": (-1,-1,-1, -1,-1,16), "dtype": "float16", "format": "NCDHW", "ori_shape": (-1,16,1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(16, 16)]}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NDHWC", "ori_shape": (4,),"ori_format": "NCDHW","range":[(4, 4)]},
                    {"shape": (-1,-1,-1, -1,-1,16), "dtype": "float16", "format": "NDHWC", "ori_shape": (-1,16,1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_9",
         "expect": RuntimeError,
         "support_expect": True}




ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case6)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case7)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case8)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case9)
