#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", "impl.dynamic.trans_data", "trans_data")

def gen_transdata_case(dynamic_input_shapes, ori_input_shapes, dtype, srcFormat, dstFormat, 
                       case_name_val, expect):
    inputs = (
        {"shape": dynamic_input_shapes,
         "dtype": dtype,
         "ori_shape": ori_input_shapes,
         "ori_format": srcFormat,
         "format": srcFormat,
         'range': [[1, 100000]] * len(dynamic_input_shapes)},
    )
    outputs = (
        {"shape": [-1],
         "dtype": dtype,
         "ori_shape": ori_input_shapes,
         "ori_format": dstFormat,
         "format": dstFormat,
         'range': [[1, 100000]] * 1},
    )
    return {"params": [inputs[0],
                       outputs[0],
                       srcFormat,
                       dstFormat
                       ],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}

ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
                 gen_transdata_case((-1, -1, -1, -1),
                                    (1, 16, 7, 7),
                                    "float16", "NCHW", "NC1HWC0", "case_1", "success"))

ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
                 gen_transdata_case((-1, -1, -1, -1),
                                    (2, 23, 35, 3),
                                    "float16", "NHWC", "NC1HWC0", "case_2", "success"))

ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
                 gen_transdata_case((-1, -1, -1),
                                    (66, 2, 100),
                                    "float16", "ND", "FRACTAL_NZ", "case_3", "success"))

# negative case #
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (2, 2, 1, 1, 16),
                                    "float16", "NC1HWC0", "NHWC", "case_4", "success"))

ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (100, 3, 7, 16, 16),
                                    "float16", "FRACTAL_NZ", "ND", "case_5", "success"))

ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
                 gen_transdata_case((-1, -1, -1, -1),
                                    (100, 2, 16, 16),
                                    "float16", "FRACTAL_Z_3D", "NDHWC", "case_6", "success"))

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)
