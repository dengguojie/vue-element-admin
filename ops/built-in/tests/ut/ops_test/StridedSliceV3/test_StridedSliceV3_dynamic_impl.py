#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("StridedSliceV3", "impl.dynamic.strided_slice_v3", "strided_slice_v3")


def gen_concat_case(shape, dtype, case_name_val, expect, input_format="ND"):
    input_x = {"shape": shape, "dtype": dtype,
               "ori_shape": shape,
               "ori_format": input_format, "format": input_format,
               'range': [[10, 10],] * len(shape)}

    begin = {"shape": (len(shape),), "dtype": "int32",
             "ori_shape": shape,
             "ori_format": input_format, "format": input_format,
             'range': [[2,2]]}    
    end = {"shape": (len(shape),), "dtype": "int32",
             "ori_shape": shape,
             "ori_format": input_format, "format": input_format,
             'range': [[2, 2]]}
    axes = {"shape": (len(shape),), "dtype": "int32",
             "ori_shape": shape,
             "ori_format": input_format, "format": input_format,
             'range': [[2, 2]]}
    strides = {"shape": (len(shape),), "dtype": "int32",
             "ori_shape": shape,
             "ori_format": input_format, "format": input_format,
             'range': [[2, 2]]}

    return {"params": [input_x,
                       begin,
                       end,
                       strides,
                       axes,
                       input_x],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_concat_case((10, 10), "float16", "case_1", "success"))


if __name__ == '__main__':
    ut_case.run("Ascend910A")
