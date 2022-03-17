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
                    {"shape": (-1,-1,-1, -1,-1,16), "dtype": "float16", "format": "NDHWC", "ori_shape": (-1,16,1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1,1),(1,100),(16,16)]},
                    ],
         "case_name": "BiasAdd_9",
         "expect": RuntimeError,
         "support_expect": True}


case10 = {"params": [{"shape": (-2,), "dtype": "float16", "format": "NCDHW", "ori_shape": (1,16,1,1,16),"ori_format": "NCDHW","range":[(1, 1),(1, 1),(1, 1),(16, 16)]}, #x
                    {"shape": (-2,), "dtype": "float16", "format": "NDHWC", "ori_shape": (4,),"ori_format": "NCDHW","range":[(4,4)]},
                    {"shape": (-2,), "dtype": "float16", "format": "NDHWC", "ori_shape": (1,16,1,1,16),"ori_format": "NCDHW","range":[(1, 1),(1, 1),(1, 1),(16, 16)]},
                    ],
         "case_name": "BiasAdd_10",
         "expect": RuntimeError,
         "support_expect": True}

case11 = {"params": [{"shape": (1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,16),"ori_format": "NHWC","range":[(1, 1),(1, 1),(1, 1),(16, 16)]}, #x
                     {"shape": (4,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,),"ori_format": "NHWC","range":[(4, 4)]},
                     {"shape": (1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,16),"ori_format": "NHWC","range":[(1, 1),(1, 100),(1, 100),(16, 16)]}
                    ],
          "case_name": "BiasAdd_11",
          "expect": RuntimeError,
          "support_expect": True}

case12 = {"params": [{"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0","range":[(1,1),(1, 1),(1, 1),(1, 1),(16, 16)]}, #x
                     {"shape": (4,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,),"ori_format": "NHWC","range":[(4, 4)]},
                     {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,16),"ori_format": "NC1HWC0","range":[(1, 1),(1, 1),(1,1),(1, 100),(16, 16)]}
                    ],
          "case_name": "BiasAdd_12",
          "expect": RuntimeError,
          "support_expect": True}

case13 = {"params": [{"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,1,-1,16),"ori_format": "NCHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}, #x
                     {"shape": (4,), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4,),"ori_format": "NCHW","range":[(4, 4)]},
                     {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,1,-1,16),"ori_format": "NCHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}
                    ],
          "case_name": "BiasAdd_13",
          "expect": RuntimeError,
          "support_expect": True}
          
case14 = {"params": [{"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,-1,-1,16),"ori_format": "NCHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}, #x
                     {"shape": (4,), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (4,),"ori_format": "NCHW","range":[(4, 4)]},
                     {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,-1,-1,16),"ori_format": "NCHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}
                    ],
          "case_name": "BiasAdd_14",
          "expect": RuntimeError,
          "support_expect": True}

case15 = {"params": [{"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NDHWC", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NDHWC","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}, #x
                    {"shape": (4,), "dtype": "float16", "format": "NDHWC", "ori_shape": (4,),"ori_format": "NCDHW","range":[(4, 4)]},
                    {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NDHWC", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NDHWC","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_15",
         "expect": RuntimeError,
         "support_expect": True}
   
case16 = {"params": [{"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NDHWC", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NDHWC","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}, #x
                    {"shape": (16,), "dtype": "float16", "format": "NDHWC", "ori_shape": (16,),"ori_format": "NDHWC","range":[(16, 16)]},
                    {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NDHWC", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NDHWC","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_16",
         "expect": "success",
         "support_expect": True}

case17 = {"params": [{"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NCDHW", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}, #x
                    {"shape": (16,), "dtype": "float16", "format": "NCDHW", "ori_shape": (16,),"ori_format": "NCDHW","range":[(16, 16)]},
                    {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NCDHW", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_17",
         "expect": RuntimeError,
         "support_expect": True}

case18 = {"params": [{"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NCDHW", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}, #x
                    {"shape": (1,), "dtype": "float16", "format": "NCDHW", "ori_shape": (1,),"ori_format": "NCDHW","range":[(1,1)]},
                    {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NCDHW", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_18",
         "expect": "success",
         "support_expect": True}

case19 = {"params": [{"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(16, 16),(1, 100),(1, 100),(16, 16)]}, #x
                    {"shape": (1,), "dtype": "float16", "format": "NCDHW", "ori_shape": (1,),"ori_format": "NCDHW","range":[(1, 1)]},
                    {"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(16, 16),(1, 100),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_19",
         "expect": RuntimeError,
         "support_expect": True}  

case20 = {"params": [{"shape": (-1,1,-1,-1,1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,1,-1,-1,16),"ori_format": "NDHWC","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]}, #x
                    {"shape": (1,), "dtype": "float16", "format": "NDHWC", "ori_shape": (1,),"ori_format": "NDHWC","range":[(1, 1)]},
                    {"shape": (-1,1,-1,-1,1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,1,-1,-1,16),"ori_format": "NDHWC","range":[(1, 100),(1, 1),(1, 100),(1, 100),(16, 16)]},
                    ],
         "case_name": "BiasAdd_20",
         "expect": RuntimeError,
         "support_expect": True}         

case21 = {"params": [{"shape": (-1,1,-1,-1,1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(1, 1),(16, 16)]}, #x
                    {"shape": (1,), "dtype": "float16", "format": "NCDHW", "ori_shape": (1,),"ori_format": "NCDHW","range":[(1, 1)]},
                    {"shape": (-1,1,-1,-1,1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(1, 1),(16, 16)]},
                    ],
         "case_name": "BiasAdd_21",
         "expect": RuntimeError,
         "support_expect": True} 

case22 = {"params": [{"shape": (-1,1,-1,-1,1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(1, 1),(16, 16)]}, #x
                    {"shape": (16,), "dtype": "float16", "format": "NCDHW", "ori_shape": (16,),"ori_format": "NCDHW","range":[(16, 16)]},
                    {"shape": (-1,1,-1,-1,1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (-1,16,-1,-1,16),"ori_format": "NCDHW","range":[(1, 100),(1, 1),(1, 100),(1, 100),(1, 1),(16, 16)]},
                    ],
         "case_name": "BiasAdd_22",
         "expect": "success",
         "support_expect": True}   
         
case23 = {"params": [{"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND","range":[(16, 16)]}, #x
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND","range":[(16, 16)]},
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND","range":[(16, 16)]},
                    ],
         "case_name": "BiasAdd_23",
         "expect": RuntimeError,
         "support_expect": True}    

case24 = {"params": [{"shape": (-1,1,16), "dtype": "float16", "format": "ND", "ori_shape": (-1,1,16),"ori_format": "ND","range":[(1, 1),(1, 1),(16, 16)]}, #x
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 1)]},
                    {"shape": (-1,1,16), "dtype": "float16", "format": "ND", "ori_shape": (-1,1,16),"ori_format": "ND","range":[(1, 1),(1, 1),(16, 16)]},
                    ],
         "case_name": "BiasAdd_24",
         "expect": RuntimeError,
         "support_expect": True}

case25 = {"params": [{"shape": (16), "dtype": "float16", "format": "ND", "ori_shape": (16),"ori_format": "ND","range":[(16, 16)]}, #x
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND","range":[(1, 1)]},
                    {"shape": (16), "dtype": "float16", "format": "ND", "ori_shape": (16),"ori_format": "ND","range":[(16, 16)]},
                    "NCHW"
                    ],
         "case_name": "BiasAdd_25",
         "expect": RuntimeError,
         "support_expect": True}

case26 = {"params": [{"shape": (-1,1,16), "dtype": "float16", "format": "ND", "ori_shape": (-1,1,16),"ori_format": "ND","range":[(1, 1),(1, 1),(16, 16)]}, #x
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND","range":[(16, 16)]},
                    {"shape": (-1,1,16), "dtype": "float16", "format": "ND", "ori_shape": (-1,1,16),"ori_format": "ND","range":[(1, 1),(1, 1),(16, 16)]},
                    "NCHW"
                    ],
         "case_name": "BiasAdd_26",
         "expect": RuntimeError,
         "support_expect": True}

case27 = {"params": [{"shape": (-2,), "dtype": "float16", "format": "NCDHW", "ori_shape": (1,1,1,1,16),"ori_format": "NCDHW","range":[(1, 1),(1, 1),(1, 1),(1, 1),(16, 16)]}, #x
                    {"shape": (-2,), "dtype": "float16", "format": "NCDHW", "ori_shape": (16,),"ori_format": "ND","range":[(16, 16)]},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NCDHW", "ori_shape": (1,1,1,1,16),"ori_format": "NCDHW","range":[(1, 1),(1, 1),(1, 1),(1, 1),(16, 16)]},
                    "NHWC"
                    ],
         "case_name": "BiasAdd_27",
         "expect": "success",
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
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case10)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case11)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case12)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case13)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case14)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case15)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case16)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case17)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case18)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case19)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case20)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case21)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case22)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case23)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case24)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case25)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case26)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case27)

