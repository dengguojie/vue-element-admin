#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SmoothL1LossV2", "impl.dynamic.smooth_l1_loss_v2", "smooth_l1_loss_v2")

# reduction = none
case1 = {"params": [{"shape": (-1,-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (-1,-1,-1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None)]},
                    {"shape": (-1,-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (-1,-1,-1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None)]},
                    {"shape": (-1,-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (-1,-1,-1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None)]},
                    1.0, "none"
                    ],
         "case_name": "smooth_l1_loss_v2_dim3_none_1",
         "expect": "success",
         "support_expect": True}
case2 = {"params": [{"shape": (-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (-1,-1),"ori_format": "ND", "range":[(1,None),(1,None)]},
                    {"shape": (-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (-1,-1),"ori_format": "ND", "range":[(1,None),(1,None)]},
                    {"shape": (-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (-1,-1),"ori_format": "ND", "range":[(1,None),(1,None)]},
                    1.0, "none"
                    ],
         "case_name": "smooth_l1_loss_v2_dim2_none_2",
         "expect": "success",
         "support_expect": True}
case3 = {"params": [{"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1, -1, -1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1, -1, -1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1, -1, -1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    1.0, "none"
                    ],
         "case_name": "smooth_l1_loss_v2_dim4_none_3",
         "expect": "success",
         "support_expect": True}
# reduction = sum
case4 = {"params": [{"shape": (-1,-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (-1,-1,-1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None)]},
                    {"shape": (-1,-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (-1,-1,-1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,1)]},
                    1.0, "sum"
                    ],
         "case_name": "smooth_l1_loss_v2_dim3_sum_4",
         "expect": "success",
         "support_expect": True}
case5 = {"params": [{"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,1)]},
                    1.0, "sum"
                    ],
         "case_name": "smooth_l1_loss_v2_dim4_sum_5",
         "expect": "success",
         "support_expect": True}
case6 = {"params": [{"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND", "range":[(1,None),(1,None)]},
                    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND", "range":[(1,None),(1,None)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,1)]},
                    1.0, "sum"
                    ],
         "case_name": "smooth_l1_loss_v2_dim2_sum_6",
         "expect": "success",
         "support_expect": True}
# reduction = mean
case7 = {"params": [{"shape": (-1,-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (-1,-1,-1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None)]},
                    {"shape": (-1,-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (-1,-1,-1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,1)]},
                    1.0, "mean"
                    ],
         "case_name": "smooth_l1_loss_v2_dim3_mean_7",
         "expect": "success",
         "support_expect": True}

case8 = {"params": [{"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1),"ori_format": "ND", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,1)]},
                    ],
         "case_name": "smooth_l1_loss_v2_dim4_mean_8",
         "expect": "success",
         "support_expect": True}
case9 = {"params": [{"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND", "range":[(1,None),(1,None)]},
                    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND", "range":[(1,None),(1,None)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,1)]}
                    ],
         "case_name": "smooth_l1_loss_v2_dim2_mean_9",
         "expect": "success",
         "support_expect": True}
# format = nchw
case10 = {"params": [{"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1, -1),"ori_format": "NCHW", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1, -1),"ori_format": "NCHW", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1, -1),"ori_format": "NCHW", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    1.0, "none"
                    ],
         "case_name": "smooth_l1_loss_v2_nchw_none_10",
         "expect": "success",
         "support_expect": True}
case11 = {"params": [{"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NCHW", "ori_shape": (-1, -1, -1, -1),"ori_format": "NCHW", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NCHW", "ori_shape": (-1, -1, -1, -1),"ori_format": "NCHW", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1,), "dtype": "float32", "format": "NCHW", "ori_shape": (-1,),"ori_format": "NCHW", "range":[(1,1)]},
                    1.0, "sum"
                    ],
         "case_name": "smooth_l1_loss_v2_nchw_sum_11",
         "expect": "success",
         "support_expect": True}
case12 = {"params": [{"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NCHW", "ori_shape": (-1, -1, -1, -1),"ori_format": "NCHW", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NCHW", "ori_shape": (-1, -1, -1, -1),"ori_format": "NCHW", "range":[(1,None),(1,None),(1,None),(1,None)]},
                    {"shape": (-1,), "dtype": "float32", "format": "NCHW", "ori_shape": (-1,),"ori_format": "NCHW", "range":[(1,1)]},
                    ],
         "case_name": "smooth_l1_loss_v2_nchw_mean_12",
         "expect": "success",
         "support_expect": True}
# format = nc1hwc0
case13 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, -1, -1, -1, 16),"ori_format": "NC1HWC0", "range":[(1,None),(1,None),(1,None),(1,None),(16,16)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, -1, -1, -1, 16),"ori_format": "NC1HWC0", "range":[(1,None),(1,None),(1,None),(1,None),(16,16)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, -1, -1, -1, 16),"ori_format": "NC1HWC0", "range":[(1,None),(1,None),(1,None),(1,None),(16,16)]},
                    1.0, "none"
                    ],
         "case_name": "smooth_l1_loss_v2_nc1hwc0_none_13",
         "expect": "success",
         "support_expect": True}
case14 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, -1, -1, -1, 16),"ori_format": "NC1HWC0", "range":[(1,None),(1,None),(1,None),(1,None),(16,16)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, -1, -1, -1, 16),"ori_format": "NC1HWC0", "range":[(1,None),(1,None),(1,None),(1,None),(16,16)]},
                    {"shape": (-1,), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (-1,),"ori_format": "NC1HWC0", "range":[(1,1)]},
                    1.0, "sum"
                    ],
         "case_name": "smooth_l1_loss_v2_nc1hwc0_sum_14",
         "expect": "success",
         "support_expect": True}
case15 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, -1, -1, -1, 16),"ori_format": "NC1HWC0", "range":[(1,None),(1,None),(1,None),(1,None),(16,16)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, -1, -1, -1, 16),"ori_format": "NC1HWC0", "range":[(1,None),(1,None),(1,None),(1,None),(16,16)]},
                    {"shape": (-1,), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (-1,),"ori_format": "NC1HWC0", "range":[(1,1)]},
                    ],
         "case_name": "smooth_l1_loss_v2_nc1hwc0_mean_15",
         "expect": "success",
         "support_expect": True}
# add case
ut_case.add_case(["Ascend310", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend910"], case9)
ut_case.add_case(["Ascend310", "Ascend910"], case10)
ut_case.add_case(["Ascend310", "Ascend910"], case11)
ut_case.add_case(["Ascend310", "Ascend910"], case12)
ut_case.add_case(["Ascend310", "Ascend910"], case13)
ut_case.add_case(["Ascend310", "Ascend910"], case14)
ut_case.add_case(["Ascend310", "Ascend910"], case15)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
