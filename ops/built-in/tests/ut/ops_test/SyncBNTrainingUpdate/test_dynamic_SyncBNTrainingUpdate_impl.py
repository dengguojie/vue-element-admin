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

ut_case = OpUT("SyncBNTrainingUpdate", "impl.dynamic.sync_bn_training_update", "sync_bn_training_update")

def gen_dynamic_sync_bn_training_update_case(shape, range, ori_shape, dtype_val, format, momentum, epsilon, kernel_name_val, expect):

    return {"params": [{"shape": shape, "dtype": dtype_val, "range": range, "format": format, "ori_shape": ori_shape, "ori_format": format},
                       {"shape": shape, "dtype": dtype_val, "range": range, "format": format, "ori_shape": ori_shape, "ori_format": format},
                       {"shape": shape, "dtype": dtype_val, "range": range, "format": format, "ori_shape": ori_shape, "ori_format": format},
                       momentum],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_sync_bn_training_update_case((1,3), [(1,None),(1,None)], (1,3),
                                                          "float32", "ND", 0.01, 0.00001, "dynamic_sync_bn_training_update_2D_fp32_ND", "success"))

if __name__ == "__main__":
    ut_case.run("Ascend910")
    ut_case.run("Ascend710")
    ut_case.run("Ascend310")

