#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("Transpose", "impl.dynamic.transpose", "transpose")


def gen_transpose_case(dynamic_input_shapes, ori_input_shapes, dtype, perm,
                       case_name_val, expect, input_format="ND"):
    inputs = (
        {"shape": dynamic_input_shapes,
         "dtype": dtype,
         "ori_shape": ori_input_shapes,
         "ori_format": input_format,
         "format": input_format,
         'range': [[1, 100000]] * len(dynamic_input_shapes)},
    )
    per_dict = (
        {"shape": [-1],
         "dtype": "int32",
         "ori_shape": ori_input_shapes,
         "ori_format": input_format,
         "format": input_format,
         'range': [[1, 100000]] * 1},
    )
    return {"params": [inputs[0],
                       per_dict[0],
                       inputs[0]
                       ],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
                 gen_transpose_case((-1, -1),
                                    (66, 2),
                                    "float32", (0, 1), "case_1", "success"))

# ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1),
#                                     (66, 2),
#                                     "float32", (1, 0), "case_2", "success"))

ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
                 gen_transpose_case((-1, -1, -1),
                                    (66, 2, 100),
                                    "float16", (0, 1, 2), "case_3", "success"))

# ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1, -1),
#                                     (66, 2, 100),
#                                     "float32", (0, 2, 1), "case_4", "success"))

# ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1, -1),
#                                     (66, 2, 100),
#                                     "float32", (1, 0, 2), "case_5", "success"))

# ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1, -1),
#                                     (66, 2, 100),
#                                     "float32", (1, 2, 0), "case_6", "success"))

# ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1, -1),
#                                     (66, 2, 100),
#                                     "float32", (2, 0, 1), "case_7", "success"))

ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"],
                 gen_transpose_case((-1, -1, -1, -1, -1, -1, -1, -1),
                                    (66, 2, 100),
                                    "float32", (2, 1, 0), "case_8", "success"))

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)
