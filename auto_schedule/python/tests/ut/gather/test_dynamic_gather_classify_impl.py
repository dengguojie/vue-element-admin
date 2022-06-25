# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from tbe.dsl import classify

ut_case = OpUT("gather_classify", "gather_classify.test_dynamic_gather_classify_impl")


@add_cust_test_func(ut_case)
def test_gather_classify_with_unknown_rank(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        x = {
            "shape": (-2,),
            "dtype": "float16",
            "range": [(0, None)],
            "param_name": "x"
        }

        indices = {
            "shape": (-2,),
            "dtype": "int32",
            "range": [(0, None)],
            "param_name": "indices"
        }

        real_axis = "unknown"
        batch_dims = "unknown"

        ret = classify([x, indices, real_axis, batch_dims], "gather")

        expected = [
            [
                {'shape': (0, 0, 0, 0), 'range': ((0, 0), (0, 0), (0, 0), (0, 0)), 'dtype': 'float16'}, 
                {'shape': (0, 0, 0), 'range': ((0, 0), (0, 0), (0, 0)), 'dtype': 'int32'}, 
                1, 
                1
            ], 
            [
                {
                    'dtype': 'float16', 
                    'shape': [-1, -1, -1, -1], 
                    'range': [(1, None), (1, None), [1, None], (1, None)]
                }, 
                {'dtype': 'int32', 'shape': [-1, -1], 'range': [(1, None), (1, None)]}, 
                2, 
                1
            ]
        ]

    return ret == expected


@add_cust_test_func(ut_case)
def test_gathernd_classify_with_unknown_rank(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        x = {
            "shape": (-2,),
            "dtype": "float16",
            "range": [(1, None)],
        }

        indices = {
            "shape": (-2,),
            "dtype": "int32",
            "range": [(1, None)],
        }

        batch_dims = 0

        ret = classify([x, indices, batch_dims], "gather_nd")

        expected = [
            [
                {'shape': (0, 0, 0, 0), 'range': ((0, 0), (0, 0), (0, 0), (0, 0)), 'dtype': 'float16'}, 
                {'shape': (0, 0, 0), 'range': ((0, 0), (0, 0), (0, 0)), 'dtype': 'int32'}, 
                1
            ], 
            [
                {'shape': (-1, -1), 'range': ((1, None), (1, None)), 'dtype': 'float16'}, 
                {'shape': (-1, -1, 0), 'range': ((1, None), (1, None), (0, 0)), 'dtype': 'int32'}, 
                1
            ], 
            [
                {'dtype': 'float16', 'shape': [-1, -1, -1], 'range': [(1, None), [1, None], (1, None)]}, 
                {'dtype': 'int32', 'shape': [-1, -1, 1], 'range': [(1, None), (1, None), (1, 1)]}, 
                1
            ], 
            [
                {
                    'dtype': 'float16', 
                    'shape': [-1, -1, -1, -1], 
                    'range': [(1, None), [1, None], [1, None], (1, None)]
                }, 
                {'dtype': 'int32', 'shape': [-1, -1, 2], 'range': [(1, None), (1, None), (2, 2)]}, 
                1
            ], 
            [
                {
                    'dtype': 'float16', 
                    'shape': [-1, -1, -1, -1, -1], 
                    'range': [(1, None), [1, None], [1, None], [1, None], (1, None)]
                }, 
                {'dtype': 'int32', 'shape': [-1, -1, 3], 'range': [(1, None), (1, None), (3, 3)]}, 
                1
            ], 
            [
                {
                    'dtype': 'float16', 
                    'shape': [-1, -1, -1, -1, -1, -1], 
                    'range': [(1, None), [1, None], [1, None], [1, None], [1, None], (1, None)]
                }, 
                {'dtype': 'int32', 'shape': [-1, -1, 4], 'range': [(1, None), (1, None), (4, 4)]}, 
                1
            ], 
            [
                {
                    'dtype': 'float16', 
                    'shape': [-1, -1, -1, -1, -1, -1, -1], 
                    'range': [(1, None), [1, None], [1, None], [1, None], [1, None], [1, None], (1, None)]
                }, 
                {'dtype': 'int32', 'shape': [-1, -1, 5], 'range': [(1, None), (1, None), (5, 5)]}, 
                1
            ], 
            [
                {
                    'dtype': 'float16', 
                    'shape': [-1, -1, -1, -1, -1, -1, -1, -1], 
                    'range': [(1, None), [1, None], [1, None], [1, None], [1, None], 
                              [1, None], [1, None], (1, None)]
                }, 
                {'dtype': 'int32', 'shape': [-1, -1, 6], 'range': [(1, None), (1, None), (6, 6)]}, 
                1
            ], 
            [
                {
                    'dtype': 'float16', 
                    'shape': [-1, -1, -1, -1, -1, -1, -1, -1, -1], 
                    'range': [(1, None), [1, None], [1, None], [1, None], [1, None], 
                              [1, None], [1, None], [1, None], (1, None)]
                }, 
                {'dtype': 'int32', 'shape': [-1, -1, 7], 'range': [(1, None), (1, None), (7, 7)]}, 
                1
            ], 
            [
                {
                    'dtype': 'float16', 
                    'shape': [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1], 
                    'range': [(1, None), [1, None], [1, None], [1, None], [1, None], 
                              [1, None], [1, None], [1, None], [1, None], (1, 1)]
                }, 
                {'dtype': 'int32', 'shape': [-1, -1, 8], 'range': [(1, None), (1, None), (8, 8)]}, 
                1
            ]
        ]

    return ret == expected


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        # if not k == "test_gathernd_classify_with_unknown_rank":
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
