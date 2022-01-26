#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("StridedSlice", "impl.dynamic.strided_slice", "strided_slice")


def gen_concat_case(shape, dtype, case_name_val, expect, input_format="ND"):
    input_x = {"shape": shape, "dtype": dtype,
               "ori_shape": shape,
               "ori_format": input_format, "format": input_format,
               'range': [[1, 100000]] * len(shape)}

    begin = {"shape": (len(shape),), "dtype": "int32",
             "ori_shape": shape,
             "ori_format": input_format, "format": input_format,
             'range': [[1, 100000]]}
    end = begin
    strides = begin

    return {"params": [input_x,
                       end,
                       strides,
                       input_x],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_concat_case((-1, -1), "float16", "case_1", "success"))
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_concat_case((-1, -1), "float32", "case_2", "success"))
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_concat_case((-1, -1), "int8", "case_3", "success"))
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_concat_case((-1, -1), "uint8", "case_4", "success"))
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_concat_case((-1, -1), "int32", "case_5", "success"))
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_concat_case((-2, ), "bool", "case_6", "success"))

def test_op_check_supported_1(test_arg):
    from impl.dynamic.strided_slice import check_supported
    input_x = {"shape": (-1, -1), "dtype": "float16", "ori_shape": (-1, -1), "ori_format": "ND", "format": "ND",
               'range': [[1, 100000]] * len((-1, -1))}

    begin = {"shape": (len((-1, -1)),), "dtype": "int32", "ori_shape": (-1, -1), "ori_format": "ND", "format": "ND",
             'range': [[1, 100000]]}
    end = begin
    strides = begin

    if check_supported(input_x, end, strides, input_x) != "Unknown":
        raise Exception("Failed to call check_supported in stridedslice.")

ut_case.add_cust_test_func(test_func=test_op_check_supported_1)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
