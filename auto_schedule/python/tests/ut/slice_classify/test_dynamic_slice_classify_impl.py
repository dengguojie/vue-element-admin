# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from tbe.dsl import classify
import tbe

ut_case = OpUT("slice_classify", "slice_classify.test_dynamic_slice_classify_impl")


def test_input_length_error(_):
    try:
        with tbe.common.context.op_context.OpContext("dynamic"):
            input_list = [{"shape": (5, -1,),
                           "dtype": "float16",
                           "range": [(5, 5), (1, None)]},
                          {
                              "shape": (2,),
                              "dtype": "int32",
                              "range": [(2, 2)]
                          }]
            classify(input_list, "slice", {"end_mode": "size"})
    except RuntimeError as e:
        # E60005
        return e.args[0].get("errCode") == "E90001"
    return False

def test_static_input(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        input_list = [
            {"shape": (32, 3,), "dtype": "float16", "range": [(32, 32), (3, 3)],},
            {"shape": (2,), "dtype": "int32", "range": [(2, 2)], "const_value":[0, 0]},
            {"shape": (2,), "dtype": "int32", "range": [(2, 2)], "const_value":[32, 1]},
        ]
        ins = classify(input_list, "slice", {"end_mode": "size"})
        return len(ins[0]) == 3 and ins[0][1][0] == 0 and ins[0][1][1] == 0 and ins[0][2][0] == 32 and ins[0][2][1] == 1

case_list = [
    test_input_length_error,
    test_static_input,
]

for item in case_list:
    ut_case.add_cust_test_func(test_func=item)


