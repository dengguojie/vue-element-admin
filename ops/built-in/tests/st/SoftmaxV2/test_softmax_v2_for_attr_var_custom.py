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
test_softmax_v2_for_attr_var_custom
"""
from op_test_frame.ut import OpUT
from impl.dynamic.softmax_v2 import softmax_v2


ut_case = OpUT("SoftmaxV2", "impl.dynamic.softmax_v2", "softmax_v2")


def run_compile_softmaxv2(dynamic_input_shapes, ori_input_shapes, dtype, axis,
                          case_name_val, input_format="ND"):
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

    return softmax_v2(inputs[0], outputs[0], axis, case_name_val)


def test_softmax(test_arg):
    from tbe.common.context import op_context
    with op_context.OpContext("dynamic"):
        run_compile_softmaxv2((-2,), (-2,), "float32", None, "dynamic_softmax_v2_binary")

ut_case.add_cust_test_func(test_func=test_softmax)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
