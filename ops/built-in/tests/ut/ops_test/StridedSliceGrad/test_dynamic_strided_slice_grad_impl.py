#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("DynamicStridedSliceGrad", "impl.dynamic.strided_slice_grad", "strided_slice_grad")


def gen_ssg_case(shape, begin, end, strides, dy_shape, dtype, case_name_val, expect, input_format="ND"):
    dict_shape = {"shape": (len(shape),), "dtype": "int32",
                  "ori_shape": (len(shape),),
                  "ori_format": input_format, "format": input_format,
                  'range': [[1, 500]]}
    dict_begin = {"shape": (len(begin),), "dtype": "int32",
                  "ori_shape": (len(begin),),
                  "ori_format": input_format, "format": input_format,
                  'range': [[1, 500]]}
    dict_end = {"shape": (len(end),), "dtype": "int32",
                "ori_shape": (len(end),),
                "ori_format": input_format, "format": input_format,
                'range': [[1, 500]]}
    dict_strides = {"shape": (len(strides),), "dtype": "int32",
                    "ori_shape": (len(strides),),
                    "ori_format": input_format, "format": input_format,
                    'range': [[1, 500]]}

    dict_dy = {"shape": dy_shape, "dtype": dtype,
               "ori_shape": dy_shape,
               "ori_format": input_format, "format": input_format,
               'range': [[1, 500]] * len(dy_shape)}

    dict_out = {"shape": shape, "dtype": dtype,
                "ori_shape": shape,
                "ori_format": input_format, "format": input_format,
                'range': [[1, 500]] * len(dy_shape)}

    return {"params": [dict_shape,
                       dict_begin,
                       dict_end,
                       dict_strides,
                       dict_dy,
                       dict_out],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A"],
                 gen_ssg_case((1, 300, 25), (2, 1), (5, 3), (1, 1), (1, 300, 2), "float16", "case_1",
                              "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
