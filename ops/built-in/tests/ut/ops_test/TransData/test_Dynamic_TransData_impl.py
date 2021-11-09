#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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


def gen_transdata_case_binary(dynamic_input_shapes, ori_input_shapes, dtype, srcFormat, dstFormat,
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
                       None,
                       None 
                       ],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A"],
                 gen_transdata_case((-1, -1, -1, -1),
                                    (1, 16, 7, 7),
                                    "float16", "NCHW", "NC1HWC0", "case_1", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_transdata_case((-1, -1, -1, -1),
                                    (2, 23, 35, 3),
                                    "float16", "NHWC", "NC1HWC0", "case_2", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_transdata_case((-1, -1, -1),
                                    (66, 2, 100),
                                    "float16", "ND", "FRACTAL_NZ", "case_3", "success"))

# negative case #
ut_case.add_case(["Ascend910A"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (2, 2, 1, 1, 16),
                                    "float16", "NC1HWC0", "NHWC", "case_4", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (100, 3, 7, 16, 16),
                                    "float16", "FRACTAL_NZ", "ND", "case_5", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_transdata_case((-1, -1, -1, -1),
                                    (100, 2, 16, 16),
                                    "float16", "FRACTAL_Z_3D", "NDHWC", "case_6", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (2, 23, 3, 3),
                                    "float16", "NC1HWC0", "NCHW", "case_7", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (2, 23, 35, 35, 32),
                                    "float16", "NDHWC", "FRACTAL_Z_3D", "case_28", "success"))

ut_case.add_case(["all"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (2, 3, 20, 20, 16),
                                    "float16", "NC1HWC0", "FRACTAL_Z", "case_29", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (42767, 11, 1, 11, 16, 16),
                                    "int32", "NCDHW", "NDC1HWC0", "case_30", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (2, 42767, 1, 11, 16, 32),
                                    "int8", "NCDHW", "NDC1HWC0", "case_31", "success"))

ut_case.add_case(["all"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (8, 2, 2, 16, 16, 16),
                                    "float16", "NCDHW", "NDC1HWC0", "case_32", "success"))

ut_case.add_case(["all"],
                 gen_transdata_case((-1, -1, -1, -1),
                                    (1, 1, 7, 7, 16),
                                    "float16", "NCHW", "NC1HWC0", "case_33", "success"))

ut_case.add_case(["all"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (3468, 1, 16, 16),
                                    "float16", "DHWCN", "FRACTAL_Z_3D", "case_34", "success"))

ut_case.add_case(["all"],
                 gen_transdata_case((-1, -1, -1, -1),
                                    (12482, 2, 16, 16),
                                    "float16", "HWCN", "FRACTAL_Z", "case_35", "success"))

ut_case.add_case(["all"],
                 gen_transdata_case((-1, -1, -1, -1),
                                    (1495, 5, 16, 16),
                                    "float16", "ND", "FRACTAL_Z", "case_36", "success"))

ut_case.add_case(["all"],
                 gen_transdata_case((-1,),
                                    (1, 2673, 16, 16),
                                    "float16", "ND", "FRACTAL_Z", "case_37", "success"))

ut_case.add_case(["all"],
                 gen_transdata_case((-1, -1),
                                    (80, 80, 16, 16),
                                    "float16", "ND", "FRACTAL_Z", "case_38", "success"))

ut_case.add_case(["all"],
                 gen_transdata_case((-1, -1, -1, -1),
                                    (331452, 80, 16, 16),
                                    "float16", "NCHW", "FRACTAL_Z", "case_39", "success"))

ut_case.add_case(["all"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (52948, 80, 16, 16),
                                    "float16", "NCDHW", "FRACTAL_Z_3D", "case_40", "success"))

ut_case.add_case(["all"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (52948, 80, 16, 16),
                                    "uint8", "NCDHW", "FRACTAL_Z_3D", "case_41", RuntimeError))

ut_case.add_case(["all"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (2, 11, 1, 11, 16, 16),
                                    "float16", "NCDHW", "NDC1HWC0", "case_42", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_transdata_case((-1, -1, -1, -1, -1),
                                    (2, 11, 2673, 11, 16, 16),
                                    "float32", "NCDHW", "NDC1HWC0", "case_43", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_transdata_case_binary((-2,),
                                           (-2,),
                                           "float32", "NCDHW", "NDC1HWC0", "binary_case_1", "success"))
                                    
if __name__ == '__main__':
    ut_case.run("Ascend910A")
