# Copyright 2020 Huawei Technologies Co., Ltd
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
var util module
"""

from tbe import tvm
from tbe.dsl.base.context import operation


def equals_for_variable_shape(ret, expected):
    context = operation.get_context()

    def equal_tvm_expr(a, b):
        return str(a) == b
    
    def equal_tvm_var(a, b):
        da = {}
        var_x = context.get_var(a.name)
        for k in b.keys():
            da[k] = getattr(var_x, "get_" + k)()
        return da == b

    def equal_tvm_const(a, b):
        return a.value == b

    def equal_python_const(a, b):
        return a == b

    func_map = {
        tvm.expr.Add: equal_tvm_expr,
        tvm.expr.Mul: equal_tvm_expr,
        tvm.expr.FloorDiv: equal_tvm_expr,
        tvm.expr.Var: equal_tvm_var,
        tvm.expr.IntImm: equal_tvm_const,
        int: equal_python_const,
    }

    for shape_a, shape_b in zip(ret, expected):
        if isinstance(shape_a, (list, tuple)):
            for dim_a, dim_b in zip(shape_a, shape_b):
                if not func_map.get(type(dim_a))(dim_a, dim_b):
                    return False
        else:
            if not func_map.get(type(shape_a))(shape_a, shape_b):
                    return False

    return True

