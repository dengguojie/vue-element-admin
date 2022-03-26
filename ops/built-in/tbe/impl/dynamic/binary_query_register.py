# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
binary_register.py
"""
import tbe.common.register as tbe_register
from impl.util.util_binary import binary_match


def register_binary_match(op_name):
    tbe_register.register_param_generalization(op_name)(binary_match(op_name))


register_binary_match("Add")
register_binary_match("ApplyAdamD")
register_binary_match("ApplyMomentumD")
register_binary_match("ApplyRMSPropD")
register_binary_match("Assign")
register_binary_match("AssignSub")
register_binary_match("BiasAdd")
register_binary_match("BiasAddGrad")
register_binary_match("Cast")
register_binary_match("Div")
register_binary_match("Fills")
register_binary_match("FloorDiv")
register_binary_match("FloorMod")
register_binary_match("Gather")
register_binary_match("GatherV2")
register_binary_match("Greater")
register_binary_match("GreaterEqual")
register_binary_match("L2Loss")
register_binary_match("LayerNorm")
register_binary_match("Less")
register_binary_match("Maximum")
register_binary_match("Minimum")
register_binary_match("Mul")
register_binary_match("Muls")
register_binary_match("Neg")
register_binary_match("Pad")
register_binary_match("Pow")
register_binary_match("RealDiv")
register_binary_match("ReduceAll")
register_binary_match("ReduceMean")
register_binary_match("ReduceProd")
register_binary_match("ReduceSum")
register_binary_match("Relu")
register_binary_match("ReluGrad")
register_binary_match("Rsqrt")
register_binary_match("RsqrtGrad")
register_binary_match("Sample")
register_binary_match("Select")
register_binary_match("Sigmoid")
register_binary_match("SigmoidGrad")
register_binary_match("Slice")
register_binary_match("SoftmaxV2")
register_binary_match("Sqrt")
register_binary_match("Square")
register_binary_match("SquaredDifference")
register_binary_match("StridedSlice")
register_binary_match("Sub")
register_binary_match("Tanh")
register_binary_match("Tile")
register_binary_match("Transpose")
register_binary_match("ZerosLike")
