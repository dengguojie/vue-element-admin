#!/usr/bin/env python
#-*- coding : UTF-8 -*-
import tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("FusedMulApplyMomentum", "impl.dynamic.fused_mul_apply_momentum", "fused_mul_apply_momentum")

case1 = {"params": [{"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100),]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100),]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100),]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]}],
         "case_name": "fused_mul_apply_momentum_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True
        }

ut_case.add_case("Ascend910A", case1)

if __name__ == "__main__":
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
        