#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("FusedMulApplyMomentumExtern", None, None)

case1 = {"params": [{"shape": (144, 16, 16, 16), "dtype": "float32","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (144, 16, 16, 16), "dtype": "float32","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (144, 16, 16, 16), "dtype": "float32","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (144, 16, 16, 16), "dtype": "float32","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (144, 16, 16, 16), "dtype": "float32","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)}],
         "case_name": "fused_mul_apply_momentum_extern_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}



ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)


if __name__ == '__main__':
    ut_case.run("Ascend910")
