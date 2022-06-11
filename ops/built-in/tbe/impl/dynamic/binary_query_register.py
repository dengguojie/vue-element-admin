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
register_binary_match("AddV2")
register_binary_match("ApplyAdamD")
register_binary_match("ApplyMomentumD")
register_binary_match("ApplyRMSPropD")
register_binary_match("Atan")
register_binary_match("Atan2")
register_binary_match("BesselI0e")
register_binary_match("BesselI1e")
register_binary_match("Erfc")
register_binary_match("Expint")
register_binary_match("Assign")
register_binary_match("AxpyV2")
register_binary_match("Addcmul")
register_binary_match("Addcdiv")
register_binary_match("AssignSub")
register_binary_match("Atanh")
register_binary_match("AtanGrad")
register_binary_match("BiasAdd")
register_binary_match("BiasAddGrad")
register_binary_match("Cast")
register_binary_match("DataFormatDimMap")
register_binary_match("Dawsn")
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
register_binary_match("OneHot")
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
register_binary_match("SelectV2")
register_binary_match("ThresholdV2")
register_binary_match("HardSwish")
register_binary_match("HardSwishGrad")
register_binary_match("Shrink")
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
register_binary_match("Abs")
register_binary_match("ApplyAdagradD")
register_binary_match("ApplyGradientDescent")
register_binary_match("ArgMaxV2")
register_binary_match("AssignAdd")
register_binary_match("BatchToSpaceND")
register_binary_match("BroadcastTo")
register_binary_match("Ceil")
register_binary_match("Cos")
register_binary_match("DepthToSpace")
register_binary_match("Dequantize")
register_binary_match("Diag")
register_binary_match("DiagPart")
register_binary_match("DivNoNan")
register_binary_match("Elu")
register_binary_match("EluGrad")
register_binary_match("Equal")
register_binary_match("Erf")
register_binary_match("Exp")
register_binary_match("Floor")
register_binary_match("GatherNd")
register_binary_match("GeluGrad")
register_binary_match("Inv")
register_binary_match("LeakyRelu")
register_binary_match("LeakyReluGrad")
register_binary_match("LessEqual")
register_binary_match("Log")
register_binary_match("Log1p")
register_binary_match("LogicalAnd")
register_binary_match("LogicalNot")
register_binary_match("LogicalOr")
register_binary_match("PadV2")
register_binary_match("Reciprocal")
register_binary_match("ReduceMax")
register_binary_match("ReduceMin")
register_binary_match("Relu6")
register_binary_match("Relu6Grad")
register_binary_match("ResizeNearestNeighborV2")
register_binary_match("ReverseV2")
register_binary_match("Rint")
register_binary_match("Round")
register_binary_match("ScatterAdd")
register_binary_match("ScatterNd")
register_binary_match("ScatterSub")
register_binary_match("Sign")
register_binary_match("Sin")
register_binary_match("Softplus")
register_binary_match("SpaceToBatchND")
register_binary_match("SpaceToDepth")
register_binary_match("SqrtGrad")
register_binary_match("StridedSliceGrad")
register_binary_match("TanhGrad")
register_binary_match("BNTrainingUpdate")
register_binary_match("BNTrainingUpdateV3")
register_binary_match("BNTrainingUpdateGrad")
register_binary_match("BNTrainingReduce")
register_binary_match("BNTrainingReduceGrad")
register_binary_match("SparseApplyAdagrad")
register_binary_match("LayerNormXBackpropV2")
register_binary_match("Invert")
register_binary_match("InvGrad")
