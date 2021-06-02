#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import sys
from impl.util.platform_adapter import tbe_context
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

ut_case = OpUT("AsStrided", "impl.dynamic.as_strided", "as_strided")

def calc_sizeof(x):
    dtype = x.get("dtype")
    if dtype == "int64" or dtype == "uint64":
        return 8
    if dtype == "int32" or dtype == "uint32"  or dtype == "float32":
        return 4
    if dtype == "int16" or dtype == "uint16"  or dtype == "float16":
        return 2 
    if dtype == "int8":
        return 1 
    return 0

def calc_expect_func(x, size, stride, storage_offset, actual):
    expect = np.lib.stride_tricks.as_strided(x.get("value"), size.get("value"), stride.get("value") * calc_sizeof(x))
    print("------------------actual---------------------")
    print(actual.get("value"))
    print("------------------expect---------------------")
    print(expect)
    return (expect,)

ut_case.add_precision_case(["Ascend920A","Ascend910A"],
                           {
                               "params":
                                   [
                                       {
                                           "shape": (-1, 240),
                                           "dtype": "int32",
                                           "format": "ND",
                                           "ori_shape": (100, 24),
                                           "range": ((100, 100), (240, 240),),
                                           "run_shape": (100, 240),
                                           "ori_format": "ND",
                                           "value": np.arange(100*240, dtype="int32").reshape(100,240),
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (2,),
                                           "run_shape": (2,),
                                           "dtype": "int32",
                                           "ori_shape": (2),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.array([100, 24]),
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (2,),
                                           "run_shape": (2,),
                                           "dtype": "int32",
                                           "ori_shape": (2),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.array([240, 10]),
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (1,),
                                           "run_shape": (1,),
                                           "dtype": "int32",
                                           "ori_shape": (1),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.array([0]),
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (-1, 24),
                                           "dtype": "int32",
                                           "format": "ND",
                                           "ori_shape": (100, 24),
                                           "range": ((100, 100), (24, 24),),
                                           "run_shape": (100, 24),
                                           "ori_format": "ND",
                                           "param_type": "output"
                                       }
                                   ],
                               "calc_expect_func": calc_expect_func,
                               "precision_standard": precision_info.PrecisionStandard(0, 0)
                           })

ut_case.add_precision_case(["Ascend920A","Ascend910A"],
                           {
                               "params":
                                   [
                                       {
                                           "shape": (-1, 240),
                                           "dtype": "int32",
                                           "format": "ND",
                                           "ori_shape": (100, 240),
                                           "range": ((100, 100), (240, 240),),
                                           "run_shape": (100, 240),
                                           "ori_format": "ND",
                                           "value": np.arange(100*240, dtype="int32").reshape(100,240),
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (2,),
                                           "run_shape": (2,),
                                           "dtype": "int32",
                                           "ori_shape": (2),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.array([100, 120]),
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (2,),
                                           "run_shape": (2,),
                                           "dtype": "int32",
                                           "ori_shape": (2),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.array([240, 2]),
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (1,),
                                           "run_shape": (1,),
                                           "dtype": "int32",
                                           "ori_shape": (1),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.array([0]),
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (-1, 120),
                                           "dtype": "int32",
                                           "format": "ND",
                                           "ori_shape": (100, 120),
                                           "range": ((100, 100), (120, 120),),
                                           "run_shape": (100, 120),
                                           "ori_format": "ND",
                                           "param_type": "output"
                                       }
                                   ],
                               "calc_expect_func": calc_expect_func,
                               "precision_standard": precision_info.PrecisionStandard(0, 0)
                           })

if __name__ == '__main__':
    simulator_lib_path = "/usr/local/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend920A"], simulator_mode="esl", simulator_lib_path=simulator_lib_path)
    #ut_case.run(["Ascend910A"], simulator_mode="ca", simulator_lib_path=simulator_lib_path)

