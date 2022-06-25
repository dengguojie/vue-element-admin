# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from tbe.dsl import classify

ut_case = OpUT("slice_classify", "slice_classify.test_dynamic_slice_classify_impl")


@add_cust_test_func(ut_case)
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

@add_cust_test_func(ut_case)
def test_static_input(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        input_list = [
            {"shape": (32, 3,), "dtype": "float16", "range": [(32, 32), (3, 3)],},
            {"shape": (2,), "dtype": "int32", "range": [(2, 2)], "const_value":[0, 0]},
            {"shape": (2,), "dtype": "int32", "range": [(2, 2)], "const_value":[32, 1]},
        ]
        ins = classify(input_list, "slice", {"end_mode": "size"})
        return len(ins[0]) == 3 and ins[0][1][0] == 0 and ins[0][1][1] == 0 and ins[0][2][0] == 32 and ins[0][2][1] == 1


@add_cust_test_func(ut_case)
def test_slice_classify_with_unknown_rank(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        x = {
            "shape": (-2,),
            "dtype": "float16",
            "range": [(1, None)]
        }

        begin = {
            "shape": (-2,),
            "dtype": "int32",
            "range": [(1, None)]
        }

        size = {
            "shape": (-2,),
            "dtype": "int32",
            "range": [(1, None)]
        }

        ret = classify([x, begin, size], "slice", {"end_mode": "size"})

        expected = [
            [
                {'shape': [-1], 'dtype': 'float16', 'range': [[1, None]]}, 
                {'shape': [1], 'dtype': 'int32', 'range': [[1, 1]]}, 
                {'shape': [1], 'dtype': 'int32', 'range': [[1, 1]]}
            ], 
            [
                {'shape': [-1, -1], 'dtype': 'float16', 'range': [[1, None], [1, None]]}, 
                {'shape': [2], 'dtype': 'int32', 'range': [[2, 2]]}, 
                {'shape': [2], 'dtype': 'int32', 'range': [[2, 2]]}
            ], 
            [
                {'shape': [-1, -1, -1], 'dtype': 'float16', 'range': [[1, None], [1, None], [1, None]]}, 
                {'shape': [3], 'dtype': 'int32', 'range': [[3, 3]]}, 
                {'shape': [3], 'dtype': 'int32', 'range': [[3, 3]]}
            ], 
            [
                {
                    'shape': [-1, -1, -1, -1], 
                    'dtype': 'float16', 
                    'range': [[1, None], [1, None], [1, None], [1, None]]
                }, 
                {'shape': [4], 'dtype': 'int32', 'range': [[4, 4]]}, 
                {'shape': [4], 'dtype': 'int32', 'range': [[4, 4]]}
            ], 
            [
                {
                    'shape': [-1, -1, -1, -1, -1], 
                    'dtype': 'float16', 
                    'range': [[1, None], [1, None], [1, None], [1, None], [1, None]]
                }, 
                {'shape': [5], 'dtype': 'int32', 'range': [[5, 5]]}, 
                {'shape': [5], 'dtype': 'int32', 'range': [[5, 5]]}
            ], 
            [
                {
                    'shape': [-1, -1, -1, -1, -1, -1], 
                    'dtype': 'float16', 
                    'range': [[1, None], [1, None], [1, None], [1, None], [1, None], [1, None]]
                }, 
                {'shape': [6], 'dtype': 'int32', 'range': [[6, 6]]}, 
                {'shape': [6], 'dtype': 'int32', 'range': [[6, 6]]}
            ], 
            [
                {
                    'shape': [-1, -1, -1, -1, -1, -1, -1], 
                    'dtype': 'float16', 
                    'range': [[1, None], [1, None], [1, None], [1, None], [1, None], [1, None], [1, None]]
                }, 
                {'shape': [7], 'dtype': 'int32', 'range': [[7, 7]]}, 
                {'shape': [7], 'dtype': 'int32', 'range': [[7, 7]]}
            ], 
            [
                {
                    'shape': [-1, -1, -1, -1, -1, -1, -1, -1], 
                    'dtype': 'float16', 
                    'range': [[1, None], [1, None], [1, None], [1, None], [1, None], [1, None], [1, None], [1, None]]
                }, 
                {'shape': [8], 'dtype': 'int32', 'range': [[8, 8]]}, 
                {'shape': [8], 'dtype': 'int32', 'range': [[8, 8]]}
            ], 
            [
                {'shape': [0], 'dtype': 'float16', 'range': [[0, 0]]}, 
                [0], 
                [0]
            ], 
            [
                {'shape': [-1, -1], 'dtype': 'float16', 'range': [[1, None], [1, None]]}, 
                {'shape': [2], 'dtype': 'int32', 'range': [[2, 2]], 'lr_depad': True}, 
                {'shape': [2], 'dtype': 'int32', 'range': [[2, 2]]}
            ]
        ]

    return ret == expected


@add_cust_test_func(ut_case)
def test_slice_classify_with_static(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        x = {
            "shape": (67, 1, 23, 2, 18, 29, 5),
            "dtype": "float16",
        }

        begin = {
            "shape": (7,),
            "dtype": "int32",
        }

        size = {
            "shape": (7,),
            "dtype": "int32",
        }

        ret = classify([x, begin, size], "slice", {"end_mode": "size"})

        expected = [
            [
                {'shape': [-1], 'dtype': 'float16', 'range': [[1, None]]}, 
                {'shape': [1], 'dtype': 'int32', 'range': [[1, 1]]}, 
                {'shape': [1], 'dtype': 'int32', 'range': [[1, 1]]}
            ], 
            [
                {'shape': [-1, -1], 'dtype': 'float16', 'range': [[1, None], [1, None]]}, 
                {'shape': [2], 'dtype': 'int32', 'range': [[2, 2]]}, 
                {'shape': [2], 'dtype': 'int32', 'range': [[2, 2]]}
            ], 
            [
                {'shape': [-1, -1, -1], 'dtype': 'float16', 'range': [[1, None], [1, None], [1, None]]}, 
                {'shape': [3], 'dtype': 'int32', 'range': [[3, 3]]}, 
                {'shape': [3], 'dtype': 'int32', 'range': [[3, 3]]}
            ], 
            [
                {
                    'shape': [-1, -1, -1, -1], 
                    'dtype': 'float16', 
                    'range': [[1, None], [1, None], [1, None], [1, None]]
                }, 
                {'shape': [4], 'dtype': 'int32', 'range': [[4, 4]]}, 
                {'shape': [4], 'dtype': 'int32', 'range': [[4, 4]]}
            ], 
            [
                {
                    'shape': [-1, -1, -1, -1, -1], 
                    'dtype': 'float16', 
                    'range': [[1, None], [1, None], [1, None], [1, None], [1, None]]
                }, 
                {'shape': [5], 'dtype': 'int32', 'range': [[5, 5]]}, 
                {'shape': [5], 'dtype': 'int32', 'range': [[5, 5]]}
            ], 
            [
                {
                    'shape': [-1, -1, -1, -1, -1, -1], 
                    'dtype': 'float16', 
                    'range': [[1, None], [1, None], [1, None], [1, None], [1, None], [1, None]]
                }, 
                {'shape': [6], 'dtype': 'int32', 'range': [[6, 6]]}, 
                {'shape': [6], 'dtype': 'int32', 'range': [[6, 6]]}
            ], 
            [
                {
                    'shape': [-1, -1, -1, -1, -1, -1, -1], 
                    'dtype': 'float16', 
                    'range': [[1, None], [1, None], [1, None], [1, None], [1, None], [1, None], [1, None]]
                }, 
                {'shape': [7], 'dtype': 'int32', 'range': [[7, 7]]}, 
                {'shape': [7], 'dtype': 'int32', 'range': [[7, 7]]}
            ], 
            [
                {'shape': [0], 'dtype': 'float16', 'range': [[0, 0]]}, 
                [0], 
                [0]
            ], 
            [
                {'shape': [-1, -1], 'dtype': 'float16', 'range': [[1, None], [1, None]]}, 
                {'shape': [2], 'dtype': 'int32', 'range': [[2, 2]], 'lr_depad': True}, 
                {'shape': [2], 'dtype': 'int32', 'range': [[2, 2]]}
            ]
        ]

    return ret == expected


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        # if not k == "test_slice_classify_with_static":
        #     continue

        try:
            ret = v.test_func(None)
        except Exception:
            import traceback
            print(f"\033[93mException: {k}\033[0m")
            print(traceback.format_exc())
            continue

        if ret:
            print(f"\033[92mPASS: {k}\033[0m")
        else:
            print(f"\033[91mFAIL: {k}\033[0m")
