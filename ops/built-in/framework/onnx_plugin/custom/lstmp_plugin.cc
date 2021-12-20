/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "graph/operator.h"
#include "register/register.h"
#include "json.hpp"
#include "./../onnx_common.h"

using namespace ge;
using json = nlohmann::json;
namespace domi {
Status ParseOnnxParamsLSTMP(const ge::Operator& op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LSTMP")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::LSTMP"),
                   ge::AscendString("ai.onnx::9::LSTMP"),
                   ge::AscendString("ai.onnx::10::LSTMP"),
                   ge::AscendString("ai.onnx::11::LSTMP"),
                   ge::AscendString("ai.onnx::12::LSTMP"),
                   ge::AscendString("ai.onnx::13::LSTMP")})
    .ParseParamsByOperatorFn(ParseOnnxParamsLSTMP)
    .ImplyType(ImplyType::TVM);
}  // domi