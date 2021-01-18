/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {
#define REGISTER_PARSE_PARAMS_OP(ATC_OPP)                                    \
  Status ParseParams##ATC_OPP(const Message *op_src, ge::Operator &op_dst) { \
    const ge::onnx::NodeProto *node =                                        \
        reinterpret_cast<const ge::onnx::NodeProto *>(op_src);               \
    if (node == nullptr) {                                                   \
      OP_LOGE(#ATC_OPP, "Dynamic cast op_src to NodeProto failed.");         \
      return FAILED;                                                         \
    }                                                                        \
    return SUCCESS;                                                          \
  }

#define REGISTER_CUSTOM_ONNX_OP(ONNX_OPP, ATC_OPP) \
  REGISTER_PARSE_PARAMS_OP(ATC_OPP)                \
  REGISTER_CUSTOM_OP(#ATC_OPP)                     \
      .FrameworkType(ONNX)                         \
      .OriginOpType(#ONNX_OPP)                     \
      .ParseParamsFn(ParseParams##ATC_OPP)         \
      .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Acos, Acos);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Acosh, Acosh);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Atan, Atan);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Ceil, Ceil);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Clip, Relu6);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Cos, Cos);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Cosh, Cosh);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Equal, Equal);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Floor, Floor);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Less, Less);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Neg, Neg);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Not, LogicalNot);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Pow, Pow);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Round, Round);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Erf, Erf);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Softplus, Softplus);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Softsign, Softsign);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Tanh, Tanh);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Reciprocal, Reciprocal);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Asin, Asin);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Asinh, Asinh);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Slice, StridedSliceV2);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Exp, Exp);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Sin, Sin);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Sinh, Sinh);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Atanh, Atanh);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Greater, Greater);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Or, LogicalOr);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Log, Log);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::And, LogicalAnd);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::PRelu, PRelu);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Sign, Sign);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Det, MatrixDeterminant);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::LessOrEqual, LessEqual);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Identity, Identity);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Where, SelectV2);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Range, Range);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Expand, Expand);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::ScatterND, TensorScatterUpdate);
}  // namespace domi
