#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("Mvn", None, None)

case1 = {"params": [{"shape": (2, 3, 2, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW"},
                    {"shape": (2, 3, 2, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW"}],
         "case_name": "mvn_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (2, 3, 2, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW"},
                    {"shape": (2, 3, 2, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 3, 2, 3),"ori_format": "NCHW"}],
         "case_name": "mvn_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

# ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
# ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case2)


def calc_expect_func(x, y, normalize_variance=True, across_channels=False,
                     eps=1e-9):
    shape_x = x['shape']
    if across_channels:
        axis = (1, 2, 3)
        num = shape_x[1] * shape_x[2] * shape_x[3]
    else:
        axis = (2, 3)
        num = shape_x[2] * shape_x[3]
    if num != 0:
        num_rec = 1.0/num
    mean = np.mean(x['value'], axis, keepdims=True)
    if normalize_variance:
        var = np.var(x['value'], axis, keepdims=True)
        sqrt = np.sqrt(var).astype(x['dtype'])
        sqrt_add = sqrt + eps   # according to caffe impl
        out = (x['value'] - mean) / sqrt_add
    else:
        out = x['value'] - mean
    return out

# ut_case.add_precision_case("all", {"params": [{"shape": (2,3,2,3), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,2,3),"ori_format": "NCHW", "param_type": "input"},
#                                               {"shape": (2,3,2,3), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,2,3),"ori_format": "NCHW", "param_type": "output"},
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
#                                    })
ut_case.add_precision_case("all", {"params": [{"shape": (2,4,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,4,16,16),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2,4,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,4,16,16),"ori_format": "NCHW", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (1,3,4,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,3,4,16),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1,3,4,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,3,4,16),"ori_format": "NCHW", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (11,3,4,4), "dtype": "float32", "format": "NCHW", "ori_shape": (11,3,4,4),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (11,3,4,4), "dtype": "float32", "format": "NCHW", "ori_shape": (11,3,4,4),"ori_format": "NCHW", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                   })

ut_case.add_case("all", {"params": [{"shape": (2, 4, 16, 16), "dtype": "float16", "format": "NCHW",
                         "ori_shape": (2, 4, 16, 16), "ori_format": "NCHW", "param_type": "input"},
                        {"shape": (2, 4, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 4, 16, 16),
                         "ori_format": "NCHW","param_type": "output"}, False, True],
                          "expect": "success",
                         })

def test_get_op_support_info_001(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.mvn import get_op_support_info
    te_set_version("Ascend310")
    get_op_support_info(
        {
            "shape": (11, 3, 4, 4),
            "dtype": "float32",
            "format": "NCHW",
            "ori_shape": (11, 3, 4, 4),
            "ori_format": "NCHW",
            "param_type": "input"
        }, None, True, False, 1e-9, "mvn")
    get_op_support_info(
        {
            "shape": (11, 3, 4, 4),
            "dtype": "float32",
            "format": "NCHW",
            "ori_shape": (11, 3, 4, 4),
            "ori_format": "NCHW",
            "param_type": "input"
        }, None, True, True, 1e-9, "mvn")
    get_op_support_info(
        {
            "shape": (11, 3, 4, 4),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (11, 3, 4, 4),
            "ori_format": "ND",
            "param_type": "input"
        }, None, True, False, 1e-9, "mvn")


def test_mvn_001(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.mvn import mvn
    te_set_version("SD3403")
    try:
        mvn(
            {
                "shape": (11, 3, 4, 4),
                "dtype": "float32",
                "format": "NCHW",
                "ori_shape": (11, 3, 4, 4),
                "ori_format": "NCHW",
                "param_type": "input"
            }, {
                "shape": (11, 3, 4, 4),
                "dtype": "float16",
                "format": "NCHW",
                "ori_shape": (11, 3, 4, 4),
                "ori_format": "NCHW",
                "param_type": "input"
            }, True, False, 1e-9, "mvn")
    except RuntimeError:
        pass

    try:
        mvn(
            {
                "shape": (11, 3, 4, 4),
                "dtype": "float16",
                "format": "NCHW",
                "ori_shape": (11, 3, 4, 4),
                "ori_format": "NCHW",
                "param_type": "input"
            }, {
                "shape": (11, 3, 4, 4),
                "dtype": "float16",
                "format": "NCHW",
                "ori_shape": (11, 3, 4, 4),
                "ori_format": "NCHW",
                "param_type": "input"
            }, True, False, 1e-9, "mvn")
    except RuntimeError:
        pass

ut_case.add_cust_test_func(test_func=test_get_op_support_info_001)
ut_case.add_cust_test_func(test_func=test_mvn_001)

if __name__ == '__main__':
    ut_case.run(["Ascend310"])

