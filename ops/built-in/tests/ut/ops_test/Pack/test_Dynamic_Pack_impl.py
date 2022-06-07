#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("DynamicPack", "impl.dynamic.pack", "pack")


def gen_pack_case(dynamic_input_shapes, ori_input_shapes, dtype, axis,
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


ut_case.add_case(["Ascend910A", "Ascend310", "Ascend310P3"],
                 gen_pack_case([(-1, -1), (-1, -1), (-1, -1)],
                                 [(66, 2), (66, 2), (66, 2)],
                                 "float16", -2, "case_1", "success"))

ut_case.add_case(["Ascend910A", "Ascend310", "Ascend310P3"],
                 gen_pack_case([(-1, -1), (-1, -1), (-1, -1)],
                                 [(66, 65), (66, 65), (66, 65)], "float16",
                                 -2, "case_12", "success"))

ut_case.add_case(["Ascend910A", "Ascend310", "Ascend310P3"],
                 gen_pack_case([(-1, -1)] * 1, [(2, 2)] * 1, "int64",
                                 -2, "case_13", "success"))

ut_case.add_case(["Ascend910A", "Ascend310", "Ascend310P3"],
                 gen_pack_case([(-2,)] * 1, [(2, 2)] * 1, "int64",
                               2, "case_14", "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
