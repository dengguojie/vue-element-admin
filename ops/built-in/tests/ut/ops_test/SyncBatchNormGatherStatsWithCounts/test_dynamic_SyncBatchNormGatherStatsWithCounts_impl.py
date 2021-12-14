"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dot ut case
"""
# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SyncBatchNormGatherStatsWithCounts", "impl.dynamic.sync_batch_norm_gather_stats_with_counts", "sync_batch_norm_gather_stats_with_counts")

def gen_dynamic_sync_batch_norm_gather_stats_with_counts_case(shape_1, range_1, ori_shape_1, shape_2, range_2, ori_shape_2, dtype_val, format, momentum, epsilon, kernel_name_val, expect):

    return {"params": [{"shape": shape_1, "dtype": dtype_val, "range": range_1, "format": format, "ori_shape": ori_shape_1, "ori_format": format},
                       {"shape": shape_1, "dtype": dtype_val, "range": range_1, "format": format, "ori_shape": ori_shape_1, "ori_format": format},
                       {"shape": shape_1, "dtype": dtype_val, "range": range_1, "format": format, "ori_shape": ori_shape_1, "ori_format": format},
                       {"shape": shape_1, "dtype": dtype_val, "range": range_1, "format": format, "ori_shape": ori_shape_1, "ori_format": format},
                       {"shape": shape_2, "dtype": dtype_val, "range": range_2, "format": format, "ori_shape": ori_shape_2, "ori_format": format},
                       {"shape": shape_2, "dtype": dtype_val, "range": range_2, "format": format, "ori_shape": ori_shape_2, "ori_format": format},
                       {"shape": shape_2, "dtype": dtype_val, "range": range_2, "format": format, "ori_shape": ori_shape_2, "ori_format": format},
                       {"shape": shape_2, "dtype": dtype_val, "range": range_2, "format": format, "ori_shape": ori_shape_2, "ori_format": format},
                       momentum, epsilon],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_sync_batch_norm_gather_stats_with_counts_case((4,3), [(1,None),(1,None)], (4,3), (1,3), [(1,None),(1,None)], (1,3),
                                                                           "float32", "ND", 0.01, 0.00001, "dynamic_sync_batch_norm_gather_stats_with_counts_2D_fp32_ND", "success"))

if __name__ == "__main__":
    ut_case.run("Ascend910")
    ut_case.run("Ascend710")
    ut_case.run("Ascend310")