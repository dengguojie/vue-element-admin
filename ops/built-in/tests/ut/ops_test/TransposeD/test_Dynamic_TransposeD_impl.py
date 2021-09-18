#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("DynamicTranspose", "impl.dynamic.transpose_d", "transpose_d")


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

    return {"params": [inputs[0],
                       inputs[0],
                       perm],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_transpose_case((-1, -1),
                                    (66, 2),
                                    "float32", (0, 1), "case_1", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1),
#                                     (66, 2),
#                                     "float32", (1, 0), "case_2", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1, -1),
#                                     (66, 2, 100),
#                                     "float32", (0, 1, 2), "case_3", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1, -1),
#                                     (66, 2, 100),
#                                     "float32", (0, 2, 1), "case_4", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1, -1),
#                                     (66, 2, 100),
#                                     "float32", (1, 0, 2), "case_5", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1, -1),
#                                     (66, 2, 100),
#                                     "float32", (1, 2, 0), "case_6", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1, -1),
#                                     (66, 2, 100),
#                                     "float32", (2, 0, 1), "case_7", "success"))

# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
#                  gen_transpose_case((-1, -1, -1),
#                                     (66, 2, 100),
#                                     "float32", (2, 1, 0), "case_8", "success"))
case1 = {'params': [{'shape': (-1, -1), 'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND', 'format': 'ND','range': [[1, 100000], [1, 100000]]},
                   {'shape': (-1, -1),'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND','format': 'ND','range': [[1, 100000], [1, 100000]]},
                   (0, 3)],
                   'case_name': 'case_1',
                   'expect': 'failed',
                   'support_expect': True}
case2 = {'params': [{'shape': (-1, -1), 'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND', 'format': 'ND','range': [[1, 100000], [1, 100000]]},
                   {'shape': (-1, -1),'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND','format': 'ND','range': [[1, 100000], [1, 100000]]},
                   (0, 1, 2)],
                   'case_name': 'case_2',
                   'expect': 'success',
                   'support_expect': True}
case3 = {'params': [{'shape': (-1, -1), 'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND', 'format': 'ND','range': [[1, 100000], [1, 100000]]},
                   {'shape': (-1, -1),'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND','format': 'ND','range': [[1, 100000], [1, 100000]]},
                   (1, 0, 2)],
                   'case_name': 'case_3',
                   'expect': 'failed',
                   'support_expect': True}
case4 = {'params': [{'shape': (-1, -1), 'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND', 'format': 'ND','range': [[1, 100000], [1, 100000]]},
                   {'shape': (-1, -1),'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND','format': 'ND','range': [[1, 100000], [1, 100000]]},
                   (2, 1, 0)],
                   'case_name': 'case_4',
                   'expect': 'success',
                   'support_expect': True}
case5 = {'params': [{'shape': (-1, -1), 'dtype': 'float16','ori_shape': (66, 2),'ori_format': 'ND', 'format': 'ND','range': [[1, 100000], [1, 100000]]},
                   {'shape': (-1, -1),'dtype': 'float16','ori_shape': (66, 2),'ori_format': 'ND','format': 'ND','range': [[1, 100000], [1, 100000]]},
                   (2, 1, 0)],
                   'case_name': 'case_5',
                   'expect': 'failed',
                   'support_expect': True}
case6 = {'params': [{'shape': (-1, -1), 'dtype': 'int64','ori_shape': (66, 2),'ori_format': 'ND', 'format': 'ND','range': [[1, 100000], [1, 100000]]},
                   {'shape': (-1, -1),'dtype': 'int8','ori_shape': (66, 2),'ori_format': 'ND','format': 'ND','range': [[1, 100000], [1, 100000]]},
                   (2, 1, 0)],
                   'case_name': 'case_6',
                   'expect': 'failed',
                   'support_expect': True}
case7 = {'params': [{'shape': (-1, -1), 'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND', 'format': 'ND','range': [[1, 100000], [1, 100000]]},
                   {'shape': (-1, -1),'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND','format': 'ND','range': [[1, 100000], [1, 100000]]},
                   (2, 1, 0, 0)],
                   'case_name': 'case_7',
                   'expect': 'failed',
                   'support_expect': True}
case8 = {'params': [{'shape': (-1, -1), 'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND', 'format': 'ND','range': [[1, 100000], [1, 100000]]},
                   {'shape': (-1, -1),'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND','format': 'ND','range': [[1, 100000], [1, 100000]]},
                   (0, 2, 1)],
                   'case_name': 'case_8',
                   'expect': 'failed',
                   'support_expect': True}
case9 = {'params': [{'shape': (-1, -1), 'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND', 'format': 'ND','range': [[1, 100000], [1, 100000]]},
                   {'shape': (-1, -1),'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND','format': 'ND','range': [[1, 100000], [1, 100000]]},
                   (1, 2, 0)],
                   'case_name': 'case_9',
                   'expect': 'failed',
                   'support_expect': True}
case10 = {'params': [{'shape': (-1, -1), 'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND', 'format': 'ND','range': [[1, 100000], [1, 100000]]},
                   {'shape': (-1, -1),'dtype': 'float32','ori_shape': (66, 2),'ori_format': 'ND','format': 'ND','range': [[1, 100000], [1, 100000]]},
                   (2, 0, 1)],
                   'case_name': 'case_10',
                   'expect': 'success',
                   'support_expect': True}
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],case1)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],case2)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],case3)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],case4)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],case5)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],case6)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],case7)
# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],case8)
# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],case9)
# ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],case10)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
