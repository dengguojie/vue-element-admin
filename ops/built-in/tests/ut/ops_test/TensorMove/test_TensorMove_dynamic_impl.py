#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT


ut_case = OpUT("TensorMove", "impl.dynamic.tensor_move", "tensor_move")


case1 = {"params": [{"shape": (-1, 8, 5),
                     "dtype": "float16",
                     "format": "ND",
                     "ori_shape": (16, 8, 5),
                     "ori_format": "ND",
                     "range":[(15,16),(8,8),(5,5)]},
                    {"shape": (-1, 8, 5),
                     "dtype": "float16",
                     "format": "ND",
                     "ori_shape":(16, 8, 5),
                     "ori_format": "ND",
                     "range":[(15,16),(8,8),(5,5)]},
                    ],
         "case_name": "TensorMove_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1,),
                     "dtype": "float32",
                     "format": "ND",
                     "ori_shape": (16,),
                     "ori_format": "ND",
                     "range":[(15,16)]},
                    {"shape": (16,),
                     "dtype": "float32",
                     "format": "ND",
                     "ori_shape":(16,),
                     "ori_format": "ND",
                     "range":[(16,16)]},
                    ],
         "case_name": "TensorMove_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, -1, 5, 3),
                     "dtype": "int32",
                     "format": "ND",
                     "ori_shape": (5, 5, 5, 3),
                     "ori_format": "ND",
                     "range":[(1,10),(1,10),(5,5),(3,3)]},
                    {"shape": (5, 5, 5, 3),
                     "dtype": "int32",
                     "format": "ND",
                     "ori_shape":(5, 5, 5, 3),
                     "ori_format": "ND",
                     "range":[(1,10),(1,10),(5,5),(3,3)]},
                    ],
         "case_name": "TensorMove_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (-1, 8, 5),
                     "dtype": "int8",
                     "format": "ND",
                     "ori_shape": (16, 8, 5),
                     "ori_format": "ND",
                     "range":[(15,16),(8,8),(5,5)]},
                    {"shape": (16, 8, 5),
                     "dtype": "int8",
                     "format": "ND",
                     "ori_shape":(16, 8, 5),
                     "ori_format": "ND",
                     "range":[(16,16),(8,8),(5,5)]},
                    ],
         "case_name": "TensorMove_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (-1, -1, -1, -1, -1, -1,),
                     "dtype": "uint8",
                     "format": "ND",
                     "ori_shape": (2, 8, 5, 10, 19, 48),
                     "ori_format": "ND",
                     "range":[(1,16),(1,10),(1,10),(1,20),(10,50)]},
                    {"shape": (2, 8, 5, 10, 19, 48),
                     "dtype": "uint8",
                     "format": "ND",
                     "ori_shape": (2, 8, 5, 10, 19, 48),
                     "ori_format": "ND",
                     "range":[(1,16),(1,10),(1,10),(1,20),(10,50)]},
                    ],
         "case_name": "TensorMove_5",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case3)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case4)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case5)
