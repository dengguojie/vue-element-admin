#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("Transpose", "impl.dynamic.transpose", "transpose")

def calc_expect_func(x, perm, y):
    x_val = x.get("value")
    p_val = perm.get("value")
    y_val = np.transpose(x_val, p_val)
    print("------------------expect---------------------")
    print(y.get("value"))
    print("------------------actual---------------------")
    print(y_val)
    return (y_val,)

def gen_transpose_case(dynamic_input_shapes, ori_input_shapes, dtype, perm_shape, case_name_val, expect, input_format="ND"):
    inputs = (
        {"shape": dynamic_input_shapes, "dtype": dtype, "ori_shape": ori_input_shapes, "ori_format": input_format, "format": input_format,
         'range': [[1, 100000]] * len(dynamic_input_shapes)},
    )
    perm = {"dtype": dtype, "orig_shape": ori_input_shapes, "shape":perm_shape}
    return {"params": [inputs[0], perm, inputs[0]], "case_name": case_name_val, "expect": expect, "support_expect": True}

#ut_case.add_case(["Ascend910A", "Ascend310"], gen_transpose_case((-1, -1), (66, 2), "float32", (0, 1), "case_1", "success"))

def test_op_check_supported(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (-1, -1), 'shape': (2, 3), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    perm = {'ori_shape': (-1,), 'shape': (2,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (-1, -1), 'shape': (3, 2), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if check_supported(input_x, perm, output_y) == False:
        raise Exception("Failed to call check_supported in Transpose.")

def test_op_check_supported_in_white_list_return_false(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    perm = {'ori_shape': (2,), 'shape': (2,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if check_supported(input_x, perm, output_y) == False:
        raise Exception("1024,1024,fp16 in white list, should return True, then call transpose instead of transpose_d")

def test_op_check_supported_not_in_white_list_return_true(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (1234, 4321), 'shape': (1234, 4321), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    perm = {'ori_shape': (2,), 'shape': (2,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (4321, 1234), 'shape': (4321, 1234), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if check_supported(input_x, perm, output_y) == True:
        raise Exception("1234,4321,fp16 not in white list, should return False")

def test_op_check_supported_dtype_not_in_white_list_return_true(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'int8'}
    perm = {'ori_shape': (2,), 'shape': (2,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'int8'}
    if check_supported(input_x, perm, output_y) == True:
        raise Exception("1024,1024,int8 not in white list, should return False")

def test_op_check_cpu_false(test_arg):
    from impl.dynamic.transpose import by_dynamic_static_union_version 
    input_x = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if by_dynamic_static_union_version(input_x, 1) == True:
        raise Exception("lhisi not support, should return False")

ut_case.add_cust_test_func(test_func=test_op_check_supported)
ut_case.add_cust_test_func(test_func=test_op_check_supported_in_white_list_return_false)
ut_case.add_cust_test_func(test_func=test_op_check_supported_not_in_white_list_return_true)
ut_case.add_cust_test_func(test_func=test_op_check_supported_dtype_not_in_white_list_return_true)


ut_case.add_precision_case("Ascend910A",
                           {
                               "params":
                                   [
                                       {
                                           "shape": (-1, 2000),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (-1, 2000),
                                           "range": ((1000, 1000), (2000, 2000),),
                                           "run_shape": (1000, 2000),
                                           "ori_format": "ND",
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (2,),
                                           "run_shape": (2,),
                                           "dtype": "int32",
                                           "ori_shape": (2),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.arange(1, -1, -1), # [1,0]
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (2000, -1),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (2000, -1),
                                           "range": ((2000, 2000), (1000, 1000),),
                                           "run_shape": (2000, 1000),
                                           "ori_format": "ND",
                                           "param_type": "output"
                                       },
                                   ],
                               "calc_expect_func": calc_expect_func,
                               "precision_standard": precision_info.PrecisionStandard(0, 0)
                           })

ut_case.add_precision_case("Ascend910A",
                           {
                               "params":
                                   [
                                       {
                                           "shape": (-1, 3, 4, 500, 601),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (-1, 3, 4, 500, 601),
                                           "range": ((2, 2), (3, 3), (4, 4), (500, 500), (601, 601),),
                                           "run_shape": (2, 3, 4, 500, 601),
                                           "ori_format": "ND",
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (5,),
                                           "run_shape": (5,),
                                           "dtype": "int32",
                                           "ori_shape": (5),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.array([0, 2, 1, 3, 4]),
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (-1, 4, 3, 500, 601),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (-1, 4, 3, 500, 601),
                                           "range": ((2, 2), (3, 3), (4, 4), (500, 500), (601, 601), ),
                                           "run_shape": (2, 4, 3, 500, 601),
                                           "ori_format": "ND",
                                           "param_type": "output"
                                       },
                                   ],
                               "calc_expect_func": calc_expect_func,
                               "precision_standard": precision_info.PrecisionStandard(0, 0)
                           })

ut_case.add_precision_case("Ascend910A",
                           {
                               "params":
                                   [
                                       {
                                           "shape": (-1, 3, 4, 5, 6),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (-1, 3, 4, 5, 6),
                                           "range": ((2, 2), (3, 3), (4, 4), (5, 5), (6, 6),),
                                           "run_shape": (2, 3, 4, 5, 6),
                                           "ori_format": "ND",
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (5,),
                                           "run_shape": (5,),
                                           "dtype": "int32",
                                           "ori_shape": (5),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.array([0, 2, 1, 3, 4]),
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (-1, 4, 3, 5, 6),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (-1, 4, 3, 5, 6),
                                           "range": ((2, 2), (4, 4), (3, 3), (5, 5), (6, 6), ),
                                           "run_shape": (2, 4, 3, 5, 6),
                                           "ori_format": "ND",
                                           "param_type": "output"
                                       },
                                   ],
                               "calc_expect_func": calc_expect_func,
                               "precision_standard": precision_info.PrecisionStandard(0, 0)
                           })

if __name__ == '__main__':
    simulator_lib_path = "/usr/local/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
#    ut_case.run("Ascend910A")
