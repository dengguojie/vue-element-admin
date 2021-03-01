#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("DynamicConcat", "impl.dynamic.concat_d", "concat_d")


def gen_concat_case(dynamic_input_shapes, ori_input_shapes, dtype, axis,
                    case_name_val, expect, input_format="ND"):
    inputs = []
    for index, shape in enumerate(dynamic_input_shapes):
        inputs.append({"shape": shape, "dtype": dtype,
                       "ori_shape": ori_input_shapes[index],
                       "ori_format": input_format, "format": input_format,
                       'range': [[1, 100000]] * len(shape)})

    return {"params": [inputs,
                       inputs[0],
                       axis],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
                                 [(66, 2), (66, 2), (66, 32)],
                                 "float16", -1, "case_1", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
#                                  [(66, 2), (66, 32), (66, 2)],
#                                  "float16",
#                                  -1, "case_2", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
#                                  [(66, 2), (66, 33), (66, 2)], "float16",
#                                  -1, "case_3", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
#                                  [(66, 2), (66, 2), (66, 2)], "float16",
#                                  -1, "case_4", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1, -1, 3, 2), (-1, -1, -1, 3, 2)],
#                                  [(10, 80, 2, 3, 2), (10, 80, 2, 3, 2)],
#                                  "float16",
#                                  -1, "case_5", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1, -1, 3, 32), (-1, -1, -1, 3, 32)],
#                                  [(10, 80, 2, 3, 32), (10, 80, 2, 4, 32)],
#                                  "float16",
#                                  -2, "case_6", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
#                                  [(66, 36), (66, 36), (66, 36)], "float16",
#                                  0, "case_7", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
#                                  [(66, 36), (66, 36), (66, 36)], "float16",
#                                  0, "case_8", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
#                                  [(66, 37), (66, 37), (66, 37)], "float16",
#                                  0, "case_9", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
#                                  [(66, 36), (66, 180), (66, 36)], "float16",
#                                  -1, "case_10", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
#                                  [(66, 64), (66, 128), (66, 64)], "float16",
#                                  -1, "case_11", "success"))

ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
                                 [(66, 65), (66, 393216), (66, 65)], "float16",
                                 -1, "case_12", "success"))

ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_concat_case([(-1, -1)] * 1, [(2, 2)] * 1, "int64",
                                 -1, "case_13", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
#                                  [(66, 65), (66, -1), (66, 65)], "float16",
#                                  0, "case_14", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, -1, -1), (-1, -1)],
#                                  [(66, 36), (66, 36, 36), (66, 36)], "float16",
#                                  -1, "err_1", "RuntimeError"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, -1), (-1, -1)],
#                                  [(66, 36), (66, 36), (66, 36)], "float16",
#                                  3, "err_2", "RuntimeError"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, 36), (-1, 37), (-1, 36)],
#                                  [(66, 36), (66, 37), (66, 36)], "float16",
#                                  0, "err_3", "RuntimeError"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, 36), (-1, 37), (-1, 36)],
#                                  [(66, 36), (66, 37), (66, 36)], "float16",
#                                  -1, "err_4", "RuntimeError", "NC1HWC0"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_concat_case([(-1, -1), (-1, 37), (-1, 36)],
#                                  [(66, -1), (66, 37), (66, 36)], "float16",
#                                  0, "err_5", "RuntimeError"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  {"params": [
#                      [{"shape": (1,), "dtype": "float16", "ori_shape": (1,),
#                        "ori_format": "ND", "format": "ND",
#                        'range': [[1, 100000], ]},
#                       {"shape": (1,), "dtype": "float32", "ori_shape": (1,),
#                        "ori_format": "ND", "format": "ND",
#                        'range': [[1, 100000], ]}],
#                      {"shape": (1,), "dtype": "float32",
#                       "ori_shape": (1,),
#                       "ori_format": "ND", "format": "ND",
#                       'range': [[1, 100000], ]},
#                      -1],
#                   "case_name": "err_6",
#                   "expect": "RuntimeError",
#                   "support_expect": True})

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)
