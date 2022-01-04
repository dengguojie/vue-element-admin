/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file npu_onnx_graph_op_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
const int64_t npuOnnxGraphOpInputPortNameLen = 4;
const int64_t npuOnnxGraphOpInputAttrNameLen = 10;
const int64_t npuOnnxGraphOpOutputPortNameLen = 6;
const int64_t npuOnnxGraphOpOutputAttrNameLen = 11;

Status AutoMappingNpuOnnxGraphOpPartitionedCall(const ge::Operator &op_src,
                                                ge::Operator &op_dest)
{
    std::vector<DynamicInputOutputInfo> value;
    DynamicInputOutputInfo input(kInput, "args",
                                 npuOnnxGraphOpInputPortNameLen,
                                 "_input_num",
                                 npuOnnxGraphOpInputAttrNameLen);
    value.push_back(input);
    DynamicInputOutputInfo output(kOutput, "output",
                                  npuOnnxGraphOpOutputPortNameLen,
                                  "_output_num",
                                  npuOnnxGraphOpOutputAttrNameLen);
    value.push_back(output);
    return AutoMappingByOpFnDynamic(op_src, op_dest, value);
}

// Register NpuOnnxGraphOp PartitionedCall to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NpuOnnxGraphOp")
    .ParseParamsByOperatorFn(AutoMappingNpuOnnxGraphOpPartitionedCall)
    .ImplyType(ImplyType::GELOCAL);
} // namespace domi
