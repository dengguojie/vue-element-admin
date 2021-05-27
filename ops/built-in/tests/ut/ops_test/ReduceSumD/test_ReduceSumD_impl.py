#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ReduceOpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = ReduceOpUT("ReduceSumD", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1,), (0,), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1,), 0, False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 1), (1,), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 1), (1,), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (101, 10241), (-1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (101, 10241), (-1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1023*255, ), (-1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1023*255, ), (-1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (1, 2), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (1, 2), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (0, 1, 2), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (0, 1, 2), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (99991, 10), (0, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (99991, 10), (0, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 99991), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 99991), (1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 99991, 10), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 99991, 10), (1, ), False)

# ============ auto gen ["Ascend910"] test cases end =================
def calc_expect_func(x, y, axis):
    res = np.sum(x['value'],axis)
    return res.astype(y['dtype'])


#ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
#                                              {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND", "param_type": "output"},
#                                              0],
#                                   "calc_expect_func": calc_expect_func,
#                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
#                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, ), "dtype": "float16", "format": "ND", "ori_shape": (16, ),"ori_format": "ND", "param_type": "output"},
                                              (1,)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (16, 4, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 4, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND", "param_type": "output"},
                                              (1, 2)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (1, 4, 1, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (1, 4, 1, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (32,), "dtype": "float16", "format": "ND", "ori_shape": (1, 4, 1, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              (0, 1, 2, 3)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (1, 11, 1, 15, 32), "dtype": "float16", "format": "ND", "ori_shape": (1, 11, 1, 15, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 15, 32), "dtype": "float16", "format": "ND", "ori_shape": (1, 15, 32),"ori_format": "ND", "param_type": "output"},
                                              (0,1)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                   })

