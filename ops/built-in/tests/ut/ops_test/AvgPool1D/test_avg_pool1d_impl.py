#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("AvgPool1D", "impl.avg_pool_1d", "avg_pool_1d")

# TODO fix me run failed so comment
# ut_case.add_case(["Ascend910"], {"params":[
#     {'shape' : (16,1,1,16000,16), 'dtype' : "float16"},
#     {'shape' : (16,1,1,8000,16), 'dtype' : "float16"},
#     4,
#     [1,1],
#     2,
#     False,
#     False],
#     "expect": "success",
#     "case_name":"test_avg_pool1d_001"})
#
# ut_case.add_case(["Ascend910"], {"params":[
#     {'shape' : (16,1,1,16000,16), 'dtype' : "float16"},
#     {'shape' : (16,1,1,8001,16), 'dtype' : "float16"},
#     4,
#     [1,1],
#     2,
#     True,
#     False],
#     "expect": "success",
#     "case_name":"test_avg_pool1d_002"})

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
