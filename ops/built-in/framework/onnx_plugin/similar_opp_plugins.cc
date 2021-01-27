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

#define REGISTER_CUSTOM_ONNX_OP(ONNX_OPP, ATC_OPP, FUN)      \
REGISTER_PARSE_PARAMS_OP(FUN)                       \
REGISTER_CUSTOM_OP(#ATC_OPP)                            \
    .FrameworkType(ONNX)                                \
    .OriginOpType(#ONNX_OPP)                            \
    .ParseParamsFn(ParseParams##FUN)                \
    .ImplyType(ImplyType::TVM);


REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Acos, Acos, AcosV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Acosh, Acosh, AcoshV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Atan, Atan, AtanV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Ceil, Ceil, CeilV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Clip, ClipByValue, ClipV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Cos, Cos, CosV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Cosh, Cosh, CoshV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Equal, Equal, EqualV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Floor, Floor, FloorV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Less, Less, LessV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Neg, Neg, NegV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Not, LogicalNot, NotV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Pow, Pow, PowV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Round, Round, RoundV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Erf, Erf, ErfV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Softplus, Softplus, SoftplusV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Softsign, Softsign, SoftsignV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Tanh, Tanh, TanhV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Reciprocal, Reciprocal, ReciprocalV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Asin, Asin, AsinV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Asinh, Asinh, AsinhV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Exp, Exp, ExpV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Sin, Sin, SinV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Sinh, Sinh, SinhV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Atanh, Atanh, AtanhV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Greater, Greater, GreaterV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Or, LogicalOr, OrV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::LogSoftmax, LogSoftmaxV2, LogSoftmaxV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Log, Log, LogV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::And, LogicalAnd, AndV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::PRelu, PRelu, PReluV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Sign, Sign, SignV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Det, MatrixDeterminant, DetV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::LSTM, DynamicRNN, LstmV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::LessOrEqual, LessEqual, LessEqualV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Identity, Identity, IdentityV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Where, SelectV2, WhereV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Range, Range, RangeV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Expand, Expand, ExpandV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::ScatterND, TensorScatterUpdate, ScatterNDV11);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::11::Dropout, Dropout, DropoutV11);

//onnx::9
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Acos, Acos, AcosV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Acosh, Acosh, AcoshV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Atan, Atan, AtanV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Ceil, Ceil, CeilV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Clip, ClipByValue, ClipV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Cos, Cos, CosV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Cosh, Cosh, CoshV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Equal, Equal, EqualV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Floor, Floor, FloorV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Less, Less, LessV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Neg, Neg, NegV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Not, LogicalNot, NotV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Pow, Pow, PowV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Round, Round, RoundV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Erf, Erf, ErfV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Softplus, Softplus, SoftplusV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Softsign, Softsign, SoftsignV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Tanh, Tanh, TanhV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Reciprocal, Reciprocal, ReciprocalV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Asin, Asin, AsinV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Asinh, Asinh, AsinhV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Exp, Exp, ExpV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Sin, Sin, SinV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Sinh, Sinh, SinhV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Atanh, Atanh, AtanhV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Greater, Greater, GreaterV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Or, LogicalOr, OrV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::LogSoftmax, LogSoftmaxV2, LogSoftmaxV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Log, Log, LogV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::And, LogicalAnd, AndV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::PRelu, PRelu, PReluV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Sign, Sign, SignV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Det, MatrixDeterminant, DetV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::LessOrEqual, LessEqual, LessEqualV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Identity, Identity, IdentityV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Where, SelectV2, WhereV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Range, Range, RangeV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Expand, Expand, ExpandV9);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::9::Dropout, Dropout, DropoutV9);

//onnx::12
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Acos, Acos, AcosV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Acosh, Acosh, AcoshV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Atan, Atan, AtanV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Ceil, Ceil, CeilV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Clip, ClipByValue, ClipV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Cos, Cos, CosV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Cosh, Cosh, CoshV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Equal, Equal, EqualV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Floor, Floor, FloorV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Less, Less, LessV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Neg, Neg, NegV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Not, LogicalNot, NotV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Pow, Pow, PowV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Round, Round, RoundV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Erf, Erf, ErfV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Softplus, Softplus, SoftplusV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Softsign, Softsign, SoftsignV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Tanh, Tanh, TanhV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Reciprocal, Reciprocal, ReciprocalV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Asin, Asin, AsinV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Asinh, Asinh, AsinhV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Exp, Exp, ExpV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Sin, Sin, SinV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Sinh, Sinh, SinhV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Atanh, Atanh, AtanhV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Greater, Greater, GreaterV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Or, LogicalOr, OrV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::LogSoftmax, LogSoftmaxV2, LogSoftmaxV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Log, Log, LogV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::And, LogicalAnd, AndV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::PRelu, PRelu, PReluV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Sign, Sign, SignV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Det, MatrixDeterminant, DetV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::LessOrEqual, LessEqual, LessEqualV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Identity, Identity, IdentityV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Where, SelectV2, WhereV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Range, Range, RangeV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Expand, Expand, ExpandV12);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::12::Dropout, Dropout, DropoutV12);

//onnx::13
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Acos, Acos, AcosV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Acosh, Acosh, AcoshV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Atan, Atan, AtanV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Ceil, Ceil, CeilV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Clip, ClipByValue, ClipV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Cos, Cos, CosV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Cosh, Cosh, CoshV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Equal, Equal, EqualV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Floor, Floor, FloorV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Less, Less, LessV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Neg, Neg, NegV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Not, LogicalNot, NotV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Pow, Pow, PowV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Round, Round, RoundV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Erf, Erf, ErfV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Softplus, Softplus, SoftplusV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Softsign, Softsign, SoftsignV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Tanh, Tanh, TanhV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Reciprocal, Reciprocal, ReciprocalV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Asin, Asin, AsinV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Asinh, Asinh, AsinhV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Exp, Exp, ExpV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Sin, Sin, SinV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Sinh, Sinh, SinhV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Atanh, Atanh, AtanhV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Greater, Greater, GreaterV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Or, LogicalOr, OrV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::LogSoftmax, LogSoftmaxV2, LogSoftmaxV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Log, Log, LogV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::And, LogicalAnd, AndV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::PRelu, PRelu, PReluV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Sign, Sign, SignV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Det, MatrixDeterminant, DetV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::LessOrEqual, LessEqual, LessEqualV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Identity, Identity, IdentityV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Where, SelectV2, WhereV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Range, Range, RangeV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Expand, Expand, ExpandV13);
REGISTER_CUSTOM_ONNX_OP(ai.onnx::13::Dropout, Dropout, DropoutV13);
}  // namespace domi
