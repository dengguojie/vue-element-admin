/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

Status AutoMappingNpuOnnxGraphOpPartitionedCall(const ge::Operator &op_src, ge::Operator &op_dest) {
  std::vector<DynamicInputOutputInfo> value;
  DynamicInputOutputInfo input(kInput, "args", 4, "_input_num", 10);
  value.push_back(input);
  DynamicInputOutputInfo output(kOutput, "output", 6, "_output_num", 11);
  value.push_back(output);
  return AutoMappingByOpFnDynamic(op_src, op_dest, value);
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NpuOnnxGraphOp")
    .ParseParamsByOperatorFn(AutoMappingNpuOnnxGraphOpPartitionedCall)
    .ImplyType(ImplyType::GELOCAL);
} // namespace domi
