#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("DynamicSlice", "impl.dynamic.slice", "slice")


def gen_concat_case(shape, dtype, case_name_val, expect, input_format="ND"):
    input_x = {"shape": shape, "dtype": dtype,
               "ori_shape": shape,
               "ori_format": input_format, "format": input_format,
               'range': [[1, 100000]] * len(shape)}

    offset = {"shape": (len(shape),), "dtype": "int32",
             "ori_shape": shape,
             "ori_format": input_format, "format": input_format,
             'range': [[1, 100000]]}
    size = offset

    return {"params": [input_x,
                       offset,
                       size,
                       input_x],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
                 gen_concat_case((-1, -1), "float16", "case_1", "success"))


if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)