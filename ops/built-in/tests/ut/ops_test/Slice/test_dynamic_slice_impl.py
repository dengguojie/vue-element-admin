#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Slice", "impl.dynamic.slice", "slice")


def gen_concat_dynamic_case(shape, dtype, case_name_val, expect, input_format="ND"):
    input_x = {"shape": shape, "dtype": dtype,
               "ori_shape": shape,
               "ori_format": input_format, "format": input_format,
               'range': tuple([(1, 100000)] * len(shape))}

    offset = {"shape": (len(shape),), "dtype": "int32",
              "ori_shape": (len(shape),),
              "ori_format": input_format, "format": input_format,
              'range': tuple([[len(shape), len(shape)]])}
    size = {"shape": (len(shape),), "dtype": "int32",
              "ori_shape": (len(shape),),
              "ori_format": input_format, "format": input_format,
              'range': tuple([[len(shape), len(shape)]])}

    return {"params": [input_x,
                       offset,
                       size,
                       input_x],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_concat_dynamic_case((-1, -1), "float16", "Slice_unknown_case_1", "success"))

# sliced case
case1 = {"params": [
    {"shape": (5, 13, 4), "dtype": "int32", "format": "NCHW", "ori_shape": (5, 13, 4), "ori_format": "NCHW"},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND", "const_value": (0, 1, 1)},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND",
     "const_value": (2, -1, -1)},
    {"shape": (2, 12, 3), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 12, 3), "ori_format": "NCHW"}
    ],
    "case_name": "SliceD_1",
    "expect": "success",
    "support_expect": True}

case2 = {"params": [
    {"shape": (65, 75), "dtype": "float32", "format": "NCHW", "ori_shape": (65, 75), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND","const_value": (13, 25)},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "const_value": (15, 33)},
    {"shape": (15, 33), "dtype": "float32", "format": "NCHW", "ori_shape": (15, 33), "ori_format": "NCHW"}
],
    "case_name": "SliceD_2",
    "expect": "success",
    "support_expect": True}

case3 = {"params": [
    {"shape": (13, 7, 5, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 3), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int64", "format": "ND", "ori_shape": (4,), "ori_format": "ND",
     "const_value": (0, 0, 0, 0)},
    {"shape": (4,), "dtype": "int64", "format": "ND", "ori_shape": (4,), "ori_format": "ND",
     "const_value": (2, 4, 3, 1)},
    {"shape": (2, 4, 3, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 4, 3, 1), "ori_format": "NCHW"}
],
    "case_name": "SliceD_3",
    "expect": "success",
    "support_expect": True}

case4 = {"params": [
    {"shape": (13, 7, 5, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 3), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int64", "format": "ND", "ori_shape": (4,), "ori_format": "ND",
     "const_value": (0, 0, 0, 0)},
    {"shape": (4,), "dtype": "int64", "format": "ND", "ori_shape": (4,), "ori_format": "ND",
     "const_value": (1, 1, 3, 1)},
    {"shape": (1, 1, 3, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 1, 3, 1), "ori_format": "NCHW"}
],
    "case_name": "SliceD_4",
    "expect": "success",
    "support_expect": True}

case5 = {"params": [
    {"shape": (13, 7, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 1, 1), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int64", "format": "ND", "ori_shape": (4,), "ori_format": "ND",
     "const_value": (0, 0, 0, 0)},
    {"shape": (4,), "dtype": "int64", "format": "ND", "ori_shape": (4,), "ori_format": "ND",
     "const_value": (2, 2, 1, 1)},
    {"shape": (2, 2, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 1, 1), "ori_format": "NCHW"}
],
    "case_name": "SliceD_5",
    "expect": "success",
    "support_expect": True}

case6 = {"params": [
    {"shape": (13, 7, 5, 5), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 5), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int64", "format": "ND", "ori_shape": (4,), "ori_format": "ND",
     "const_value": (0, 0, 0, 0)},
    {"shape": (4,), "dtype": "int64", "format": "ND", "ori_shape": (4,), "ori_format": "ND",
     "const_value": (1, 1, 1, 1)},
    {"shape": (1, 1, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 1, 1, 1), "ori_format": "NCHW"}
],
    "case_name": "SliceD_6",
    "expect": "success",
    "support_expect": True}

case7 = {"params": [
    {"shape": (2, 70000), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 70000), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "const_value": (0, 0)},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND",
     "const_value": (2, 69999)},
    {"shape": (2, 69999), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 69999), "ori_format": "NCHW"}
],
    "case_name": "SliceD_7",
    "expect": "success",
    "support_expect": True}

case8 = {"params": [
    {"shape": (7, 200, 600), "dtype": "float16", "format": "NCHW", "ori_shape": (7, 200, 600), "ori_format": "NCHW"},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND", "const_value": (1, 1, 1)},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND",
     "const_value": (3, 128, 512)},
    {"shape": (3, 128, 512), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 128, 512), "ori_format": "NCHW"}
],
    "case_name": "SliceD_8",
    "expect": "success",
    "support_expect": True}

case9 = {"params": [
    {"shape": (9, 11, 270000), "dtype": "float16", "format": "NCHW", "ori_shape": (9, 11, 270000),
     "ori_format": "NCHW"},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,),
     "ori_format": "ND", "const_value": (3, 4, 5)},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,),
     "ori_format": "ND", "const_value": (3, 5, 262144)},
    {"shape": (3, 5, 262144), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 5, 262144),
     "ori_format": "NCHW"}
],
    "case_name": "SliceD_9",
    "expect": "success",
    "support_expect": True}

case10 = {"params": [
    {"shape": (459999,), "dtype": "float16", "format": "NCHW", "ori_shape": (459999,),"ori_format": "NCHW"},
    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),
    "ori_format": "ND", "const_value": (3,)},
    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),
    "ori_format": "ND", "const_value": (458752,)},
    {"shape": (458752,), "dtype": "float16", "format": "NCHW", "ori_shape": (458752,), "ori_format": "NCHW"}
],
    "case_name": "SliceD_10",
    "expect": "success",
    "support_expect": True}

case11 = {"params": [
    {"shape": (65536, 31748), "dtype": "int64", "format": "NCHW", "ori_shape": (65536, 31748), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "const_value": (0, 0)},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND",
     "const_value": (65536, 31748)},
    {"shape": (65536, 31748), "dtype": "int64", "format": "NCHW", "ori_shape": (65536, 31748), "ori_format": "NCHW"}
],
    "case_name": "SliceD_11",
    "expect": "success",
    "support_expect": True}

case12 = {"params": [
    {"shape": (160000, 16), "dtype": "int64", "format": "NCHW", "ori_shape": (160000, 16), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "const_value": (0, 0)},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND",
     "const_value": (160000, 16)},
    {"shape": (160000, 16), "dtype": "int64", "format": "NCHW", "ori_shape": (160000, 16), "ori_format": "NCHW"}
],
    "case_name": "SliceD_12",
    "expect": "success",
    "support_expect": True}

case13 = {"params": [
    {"shape": (15, 64, 568, 568), "dtype": "float16", "format": "NCHW", "ori_shape": (15, 64, 568, 568),
     "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int64", "format": "ND", "ori_shape": (4,),
     "ori_format": "ND", "const_value": (0, 0, 0, 0)},
    {"shape": (4,), "dtype": "int64", "format": "ND", "ori_shape": (4,),
     "ori_format": "ND", "const_value": (15, 64, 392, 392)},
    {"shape": (15, 64, 392, 392), "dtype": "int64", "format": "NCHW", "ori_shape": (15, 64, 392, 392),
     "ori_format": "NCHW"}
],
    "case_name": "SliceD_13",
    "expect": "success",
    "support_expect": True}

case14 = {"params": [
    {"shape": (16000, 3121), "dtype": "float16", "format": "NCHW", "ori_shape": (16000, 3121),
     "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "const_value": (0, 3120)},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND",
     "const_value": (16000, 1)},
    {"shape": (16000, 1), "dtype": "int64", "format": "NCHW", "ori_shape": (16000, 1),
     "ori_format": "NCHW"}
],
    "case_name": "SliceD_14",
    "expect": "success",
    "support_expect": True}
case15 = {"params": [
    {"shape": (23, 4, 11, 50, 26, 13, 1, 23), "dtype": "float32", "format": "NCHW",
     "ori_shape": (23, 4, 11, 50, 26, 13, 1, 23), "ori_format": "NCHW"},
    {"shape": (8,), "dtype": "int32", "format": "ND", "ori_shape": (8,), "ori_format": "ND",
     "const_value": (0, 0, 0, 0, 0, 0, 0, 0)},
    {"shape": (8,), "dtype": "int32", "format": "ND", "ori_shape": (8,), "ori_format": "ND",
     "const_value": (-1, -1, -1, -1, -1, -1, -1, 2)},
    {"shape": (23, 4, 11, 50, 26, 13, 1, 2), "dtype": "float32", "format": "NCHW",
     "ori_shape": (23, 4, 11, 50, 26, 13, 1, 2), "ori_format": "NCHW"}
],
    "case_name": "SliceD_15",
    "expect": "success",
    "support_expect": True}

case16 = {"params": [
    {"shape": (131072, 1270), "dtype": "int16", "format": "NCHW", "ori_shape": (131072, 1270),
     "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "const_value": (0, 0)},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND",
     "const_value": (131072, 1269)},
    {"shape": (131072, 1269), "dtype": "int16", "format": "NCHW", "ori_shape": (131072, 1269),
     "ori_format": "NCHW"}
],
    "case_name": "SliceD_16",
    "expect": "success",
    "support_expect": True}

case17 = {"params": [
    {"shape": (8732, 4), "dtype": "float32", "format": "NCHW", "ori_shape": (8732, 4),
     "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "const_value": (0, 1)},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "const_value": (8732, 2)},
    {"shape": (8732, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (8732, 2),
     "ori_format": "NCHW"}
],
    "case_name": "SliceD_17",
    "expect": "success",
    "support_expect": True}

case18 = {"params": [
    {"shape": (4, 8733, 4), "dtype": "float32", "format": "NCHW", "ori_shape": (4, 8733, 4),
     "ori_format": "NCHW"},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND", "const_value": (1, 1, 1)},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND",
     "const_value": (2, 8732, 2)},
    {"shape": (2, 8732, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 8732, 2),
     "ori_format": "NCHW"}
],
    "case_name": "SliceD_18",
    "expect": "success",
    "support_expect": True}

case19 = {"params": [
    {"shape": (279424, 4), "dtype": "float32", "format": "NCHW", "ori_shape": (279424, 4),
     "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND",
     "const_value": (270001, 1)},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "const_value": (8732, 2)},
    {"shape": (8732, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (8732, 2),
     "ori_format": "NCHW"}
],
    "case_name": "SliceD_19",
    "expect": "success",
    "support_expect": True}

case20 = {"params": [
    {"shape": (10, 160), "dtype": "float16", "format": "ND", "ori_shape": (10, 160),
     "ori_format": "ND"},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "const_value": (0, 1)},
    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND", "const_value": (10, 150)},
    {"shape": (10, 150), "dtype": "float16", "format": "ND", "ori_shape": (10, 150),
     "ori_format": "ND"}
],
    "case_name": "SliceD_20",
    "expect": "success",
    "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend710"], case2)
ut_case.add_case(["Ascend910A", "Ascend710"], case3)
ut_case.add_case(["Ascend910A", "Ascend710"], case4)
ut_case.add_case(["Ascend910A", "Ascend710"], case5)
ut_case.add_case(["Ascend910A", "Ascend710"], case6)
ut_case.add_case(["Ascend910A", "Ascend710"], case7)
ut_case.add_case(["Ascend910A", "Ascend710"], case8)
ut_case.add_case(["Ascend910A", "Ascend710"], case9)
ut_case.add_case(["Ascend910A", "Ascend710"], case10)
ut_case.add_case(["Ascend910A", "Ascend710"], case11)
ut_case.add_case(["Ascend910A", "Ascend710"], case12)
ut_case.add_case(["Ascend910A", "Ascend710"], case13)
ut_case.add_case(["Ascend910A", "Ascend710"], case14)
ut_case.add_case(["Ascend910A", "Ascend710"], case15)
ut_case.add_case(["Ascend910A", "Ascend710"], case16)
ut_case.add_case(["Ascend910A", "Ascend710"], case17)
ut_case.add_case(["Ascend910A", "Ascend710"], case18)
ut_case.add_case(["Ascend910A", "Ascend710"], case19)
ut_case.add_case(["Ascend910A", "Ascend710"], case20)

# slicedv2 case
case101 = {"params": [
    {"shape": (5, 13, 4), "dtype": "int32", "format": "NCHW", "ori_shape": (5, 13, 4), "ori_format": "NCHW"},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND",
     "const_value": (2, -1, -1)},
    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND"},
    {"shape": (2, 12, 3), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 12, 3), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_1",
    "expect": "success",
    "support_expect": True}

case102 = {
    "params": [{"shape": (65, 75), "dtype": "float32", "format": "NCHW", "ori_shape": (65, 75), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (3,), "ori_format": "NCHW",
     "const_value": (15, 33)},
    {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (2,), "ori_format": "NCHW"},
    {"shape": (15, 33), "dtype": "float32", "format": "NCHW", "ori_shape": (15, 33), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_2",
    "expect": "success",
    "support_expect": True}

case103 = {"params": [
    {"shape": (13, 7, 5, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 3), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW",
     "const_value": (2, 4, 3, 1)},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW"},
    {"shape": (2, 4, 3, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 4, 3, 1), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_3",
    "expect": "success",
    "support_expect": True}

case104 = {"params": [
    {"shape": (13, 7, 5, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 3), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW",
     "const_value": (1, 1, 3, 1)},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW"},
    {"shape": (1, 1, 3, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 1, 3, 1), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_4",
    "expect": "success",
    "support_expect": True}

case105 = {"params": [
    {"shape": (13, 7, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 1, 1), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW",
     "const_value": (2, 2, 1, 1)},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW"},
    {"shape": (2, 2, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 1, 1), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_5",
    "expect": "success",
    "support_expect": True}

case106 = {"params": [
    {"shape": (13, 7, 5, 5), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 5), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW",
     "const_value": (1, 1, 1, 1)},
    {"shape": (4,), "dtype": "int32", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW"},
    {"shape": (1, 1, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 1, 1, 1), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_6",
    "expect": "success",
    "support_expect": True}

case107 = {"params": [
    {"shape": (2, 70000), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 70000), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (2,), "ori_format": "NCHW",
     "const_value": (2, 69999)},
    {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (2,), "ori_format": "NCHW"},
    {"shape": (2, 69999), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 69999), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_7",
    "expect": "success",
    "support_expect": True}

case108 = {"params": [
    {"shape": (7, 200, 600), "dtype": "float16", "format": "NCHW", "ori_shape": (7, 200, 600), "ori_format": "NCHW"},
    {"shape": (3,), "dtype": "int32", "format": "NCHW", "ori_shape": (3,), "ori_format": "NCHW",
     "const_value": (3, 128, 512)},
    {"shape": (3,), "dtype": "int32", "format": "NCHW", "ori_shape": (3,), "ori_format": "NCHW"},
    {"shape": (3, 128, 512), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 128, 512), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_8",
    "expect": "success",
    "support_expect": True}

case109 = {"params": [
    {"shape": (9, 11, 270000), "dtype": "float16", "format": "NCHW", "ori_shape": (9, 11, 270000),
     "ori_format": "NCHW"},
    {"shape": (3,), "dtype": "int32", "format": "NCHW", "ori_shape": (3,), "ori_format": "NCHW",
     "const_value": (3, 5, 262144)},
    {"shape": (3,), "dtype": "int32", "format": "NCHW", "ori_shape": (3,), "ori_format": "NCHW"},
    {"shape": (3, 5, 262144), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 5, 262144), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_9",
    "expect": "success",
    "support_expect": True}

case110 = {"params": [
    {"shape": (459999,), "dtype": "float16", "format": "NCHW", "ori_shape": (459999,), "ori_format": "NCHW"},
    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),
     "ori_format": "NCHW", "const_value": (458752,)},
    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,), "ori_format": "NCHW"},
    {"shape": (458752,), "dtype": "float16", "format": "NCHW", "ori_shape": (458752,), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_10",
    "expect": "success",
    "support_expect": True}

case111 = {"params": [
    {"shape": (65536, 31748), "dtype": "int64", "format": "NCHW", "ori_shape": (65536, 31748), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int64", "format": "NCHW", "ori_shape": (2,),
     "ori_format": "NCHW", "const_value": (65536, 31748)},
    {"shape": (2,), "dtype": "int64", "format": "NCHW", "ori_shape": (2,), "ori_format": "NCHW"},
    {"shape": (0, 0), "dtype": "int64", "format": "NCHW", "ori_shape": (0, 0), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_11",
    "expect": "success",
    "support_expect": True}

case112 = {"params": [
    {"shape": (160000, 16), "dtype": "int64", "format": "NCHW", "ori_shape": (160000, 16), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int64", "format": "NCHW", "ori_shape": (2,),
     "ori_format": "NCHW", "const_value": (160000, 16)},
    {"shape": (2,), "dtype": "int64", "format": "NCHW", "ori_shape": (2,), "ori_format": "NCHW"},
    {"shape": (160000, 16), "dtype": "int64", "format": "NCHW", "ori_shape": (160000, 16), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_12",
    "expect": "success",
    "support_expect": True}

case113 = {"params": [
    {"shape": (13, 7, 5, 5), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 5), "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int8", "format": "NCHW", "ori_shape": (4,),
     "ori_format": "NCHW", "const_value": (1, 1, 1, 1)},
    {"shape": (4,), "dtype": "int8", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW"},
    {"shape": (1, 1, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 1, 1, 1), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_13",
    "expect": RuntimeError,
    "support_expect": False}

case114 = {"params": [
    {"shape": (160000, 16), "dtype": "int32", "format": "NCHW", "ori_shape": (160000, 16), "ori_format": "NCHW"},
    {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (2,),
     "ori_format": "NCHW", "const_value": (160000, 16)},
    {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (2,), "ori_format": "NCHW"},
    {"shape": (160000, 16), "dtype": "int32", "format": "NCHW", "ori_shape": (160000, 16), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_14",
    "expect": RuntimeError,
    "support_expect": False}

case115 = {"params": [
    {"shape": (459999,), "dtype": "float16", "format": "NCHW", "ori_shape": (459999,), "ori_format": "NCHW"},
    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (3,),
     "ori_format": "NCHW", "const_value": (458752,)},
    {"shape": (3,), "dtype": "int32", "format": "NCHW", "ori_shape": (3,), "ori_format": "NCHW"},
    {"shape": (458752,), "dtype": "float16", "format": "NCHW", "ori_shape": (458752,), "ori_format": "NCHW"}
],
    "case_name": "SliceDV2_15",
    "expect": RuntimeError,
    "support_expect": False}
ut_case.add_case(["Ascend910A", "Ascend710"], case101)
ut_case.add_case(["Ascend910A", "Ascend710"], case102)
ut_case.add_case(["Ascend910A", "Ascend710"], case103)
ut_case.add_case(["Ascend910A", "Ascend710"], case104)
ut_case.add_case(["Ascend910A", "Ascend710"], case105)
ut_case.add_case(["Ascend910A", "Ascend710"], case106)
ut_case.add_case(["Ascend910A", "Ascend710"], case107)
ut_case.add_case(["Ascend910A", "Ascend710"], case108)
ut_case.add_case(["Ascend910A", "Ascend710"], case109)
ut_case.add_case(["Ascend910A", "Ascend710"], case110)
ut_case.add_case(["Ascend910A", "Ascend710"], case111)
ut_case.add_case(["Ascend910A", "Ascend710"], case112)
ut_case.add_case(["Ascend910A", "Ascend710"], case113)
ut_case.add_case(["Ascend910A", "Ascend710"], case114)
ut_case.add_case(["Ascend910A", "Ascend710"], case115)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
