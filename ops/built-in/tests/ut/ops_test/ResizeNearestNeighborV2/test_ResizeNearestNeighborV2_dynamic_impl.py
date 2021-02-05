#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ResizeNearestNeighborV2", "impl.dynamic.resize_nearest_neighbor_v2", "resize_nearest_neighbor_v2")

case1 = {"params": [{"shape": (32,1,512,512,16), "dtype": "float32", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW", "range": [(1,None), (1,None), (1,None), (1,None),(1,None)]},
                    {"shape": (32,1,512,512,16), "dtype": "int32", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW", "range": [(1,None), (1,None), (1,None), (1,None),(1,None)]},
                    {"shape": (32,1,512,512,16), "dtype": "float32", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW", "range": [(1,None), (1,None), (1,None), (1,None),(1,None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (-1,-1,-1,-1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW", "range": [(1,None), (1,None), (1,None), (1,None),(1,None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW", "range": [(1,None)]},
                    {"shape": (-1,-1,-1,-1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW", "range": [(1,None), (1,None), (1,None), (1,None),(1,None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (-1,-1,-1,-1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW", "range": [(1,None), (1,None), (1,None), (1,None),(1,None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW", "range": [(1,None)]},
                    {"shape": (-1,-1,-1,-1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (34,2,1,1,16),"ori_format": "NCHW", "range": [(1,None), (1,None), (1,None), (1,None),(1,None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)





if __name__ == '__main__':
    import te
    with te.op.dynamic():
        ut_case.run("Ascend910")

