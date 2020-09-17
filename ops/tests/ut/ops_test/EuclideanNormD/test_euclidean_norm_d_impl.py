"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

euclidean_norm_d ut case
"""
from op_test_frame.ut import ReduceOpUT

ut_case = ReduceOpUT("EuclideanNormD", None, None)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1,), (0,), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1,), 0, False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 1), (1,), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 1), (1,), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (101, 10241), (-1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (101, 10241), (-1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1023*255, ), (-1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1023*255, ), (-1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (51, 101, 1023), (1, 2), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (51, 101, 1023), (1, 2), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (51, 101, 1023), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (51, 101, 1023), (1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (51, 101, 1023), (0, 1, 2), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (51, 101, 1023), (0, 1, 2), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (99991, 10), (0, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (99991, 10), (0, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 99991), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 99991), (1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 99991, 10), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 99991, 10), (1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (33, 33, 33), (), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (5,6,3,4,8,7), (1, 3, -5), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (5,6,3,4,8,7,2), (1, 3, 4), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (5,6,3,4), (1, 2), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32", "int32"], (5,6,3,4), (0,2), False)

ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1,), (0,), True)
ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1,), 0, False)
ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1, 1), (1,), True)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1, 1), (1,), False)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (101, 10241), (-1, ), True)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (101, 10241), (-1, ), False)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1023*255, ), (-1, ), True)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1023*255, ), (-1, ), False)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (51, 101, 1023), (1, 2), True)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (51, 101, 1023), (1, 2), False)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (51, 101, 1023), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (99991, 10), (0, ), True)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1, 99991), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1, 99991), (1, ), False)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1, 99991, 10), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (1, 99991, 10), (1, ), False)
# ut_case.add_reduce_case_simple(["Ascend310"], ["float16", "int32"], (33, 33, 33), (), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "int32"], (5,6,3,4,8,7), (1, 3, -5), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "int32"], (5,6,3,4,8,7,2), (1, 3, 4), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "int32"], (5,6,3,4), (1, 2), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "int32"], (5,6,3,4), (0,2), False)

if __name__ == '__main__':
    ut_case.run("Ascend910")
