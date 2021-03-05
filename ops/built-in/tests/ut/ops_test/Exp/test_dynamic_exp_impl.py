#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from op_test_frame.ut import ElementwiseOpUT
from op_test_frame.common import precision_info

ut_case = ElementwiseOpUT("Exp", "impl.dynamic.exp", "exp")


def calc_expect_func(x, y):
    x_value = x.get("value")
    output_data = np.exp(x_value)
    return (output_data,)


# ut_case.add_precision_case("Ascend910A",
#                            {
#                                "params":
#                                    [
#                                        {
#                                            "shape": (2, -1),
#                                            "dtype": "float32",
#                                            "format": "ND",
#                                            "ori_shape": (2, -1),
#                                            "range": ((2, 2), (2, 10),),
#                                            "run_shape": (2, 10),
#                                            "ori_format": "ND",
#                                            "param_type": "input"
#                                        },
#                                        {
#                                            "shape": (2, -1),
#                                            "dtype": "float32",
#                                            "format": "ND",
#                                            "ori_shape": (2, -1),
#                                            "range": ((2, 2), (2, 10),),
#                                            "run_shape": (2, 10),
#                                            "ori_format": "ND",
#                                            "param_type": "output"
#                                        },
#                                    ],
#                                "calc_expect_func": calc_expect_func,
#                                "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001)
#                            })
#
# ut_case.add_precision_case("Ascend310",
#                            {
#                                "params":
#                                    [
#                                        {
#                                            "shape": (2, -1),
#                                            "dtype": "float32",
#                                            "format": "ND",
#                                            "ori_shape": (2, -1),
#                                            "range": ((2, 2), (2, 10),),
#                                            "run_shape": (2, 10),
#                                            "ori_format": "ND",
#                                            "param_type": "input"
#                                        },
#                                        {
#                                            "shape": (2, -1),
#                                            "dtype": "float32",
#                                            "format": "ND",
#                                            "ori_shape": (2, -1),
#                                            "range": ((2, 2), (2, 10),),
#                                            "run_shape": (2, 10),
#                                            "ori_format": "ND",
#                                            "param_type": "output"
#                                        },
#                                    ],
#                                "calc_expect_func": calc_expect_func,
#                                "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                            })
if __name__ == '__main__':
    simulator_lib_path = "/usr/local/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
