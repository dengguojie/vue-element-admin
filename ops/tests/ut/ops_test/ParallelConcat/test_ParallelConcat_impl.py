#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("ParallelConcat", None, None)

def test_parallel_concat(shape_val, dtype_val, out_shape, case_name):
    input_ = []
    for _ in range(out_shape[0]):
        dic = {'dtype': dtype_val, 'shape': shape_val, "ori_shape": shape_val, "format": "NHWC", "ori_format": "NHWC"}
        dic['dtype'] = dtype_val
        dic['shape'] = shape_val
        input_.append(dic)
    out_dic = {'dtype': dtype_val, 'shape': out_shape, "ori_shape": out_shape, "format": "NHWC", "ori_format": "NHWC"}

    return {"params": [input_, out_dic, out_shape, 2],
            "case_name": case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1 = test_parallel_concat([1, 1024], "float16", [2, 1024],"parallel_concat_1")
case2 = test_parallel_concat([1, 1024, 124], "float16", [2, 1024, 124],"parallel_concat_2")
case3 = test_parallel_concat([1, 1024, 100], "float16", [2, 1024, 100],"parallel_concat_3")
case4 = test_parallel_concat([1, 248, 1024], "float16", [2, 248, 1024],"parallel_concat_4")
case5 = test_parallel_concat([1, 17, 124, 1024], "float16", [2, 17, 124, 1024],"parallel_concat_5")
case6 = test_parallel_concat([1, 247, 1024], "float16", [2, 247, 1024],"parallel_concat_6")


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)


