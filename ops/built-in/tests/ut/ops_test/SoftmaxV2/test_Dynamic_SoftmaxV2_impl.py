# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
test_Dynamic_SoftmaxV2_impl.py
"""
import tbe
from op_test_frame.ut import OpUT
from impl.dynamic.softmax_v2 import op_select_format


ut_case = OpUT("SoftmaxV2", "impl.dynamic.softmax_v2", "softmax_v2")


def gen_softmaxv2_case(dynamic_input_shapes, ori_input_shapes, dtype, axis,
                       case_name_val, impl_mode, expect, input_format="ND"):
    inputs = (
        {"shape": dynamic_input_shapes,
         "dtype": dtype,
         "ori_shape": ori_input_shapes,
         "ori_format": input_format,
         "format": input_format,
         'range': [[1, 100000]] * len(dynamic_input_shapes)},
    )
    outputs = (
        {"shape": [-1],
         "dtype": dtype,
         "ori_shape": ori_input_shapes,
         "ori_format": input_format,
         "format": input_format,
         'range': [[1, 100000]] * 1},
    )

    return {"params": [inputs[0],
                       outputs[0],
                       axis],
            'addition_params': {'impl_mode': impl_mode},
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"],
                 gen_softmaxv2_case((-1, -1, -1),
                                    (16, 16, 16),
                                    "float16", -1, "dynamic_softmax_v2_1", "high_performance", "success"))

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"],
                 gen_softmaxv2_case((-1, -1, -1),
                                    (16, 16, 16),
                                    "float32", -1, "dynamic_softmax_v2_3", "high_performance", "success"))
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"],
                 gen_softmaxv2_case((-2,),
                                    (-2,),
                                    "float32", None, "dynamic_softmax_v2_binary", "high_performance", "success"))
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"],
                 gen_softmaxv2_case((-2,),
                                    (-2,),
                                    "float32", -1, "dynamic_softmax_v2_unknown_rank", "high_performance", "success"))
ut_case.add_case(["Ascend710", "Ascend910A"],
                 gen_softmaxv2_case((-1, -1, -1),
                                    (16, 16, 16),
                                    "float32", -1, "dynamic_softmax_v2_4", "high_precision", "success"))

def test_op_select_format(test_arg):
    op_select_format({"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                     {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                     -1)


def test_import_lib(test_arg):
    import sys
    import importlib
    importlib.reload(sys.modules.get("impl.util.norm_pattern_adapter"))
    importlib.reload(sys.modules.get("impl.util.platform_adapter"))


ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_import_lib)


if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
