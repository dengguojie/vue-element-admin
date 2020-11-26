#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TensorMove", "impl.tensor_move",
               "tensor_move")


def gen_tensor_move_case(x_shape, dtype_x, case_name_val, expect):
    return {"params": [{"shape": x_shape, "dtype": dtype_x, "ori_shape": x_shape, "ori_format": "ND", "format": "ND"},
                       {"shape": x_shape, "dtype": dtype_x, "ori_shape": x_shape, "ori_format": "ND", "format": "ND"}],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


ut_case.add_case("all",
                 gen_tensor_move_case((33,5), "float32", "valid_fp32", "success"))

ut_case.add_case("all",
                 gen_tensor_move_case((33,5,6), "int32", "valid_int32", "success"))

ut_case.add_case("all",
                 gen_tensor_move_case((33,5,6,7), "int8", "valid_int8", "success"))

ut_case.add_case("all",
                 gen_tensor_move_case((33,5,7,8,9), "float16", "valid_fp16", "success"))

ut_case.add_case("all",
                 gen_tensor_move_case((33,5,9), "uint8", "valid_uint8", "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910")
