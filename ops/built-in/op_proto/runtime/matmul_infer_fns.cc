
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
 * \file matmul_infer_fns.cc
 * \brief
 */
#include <map>
#include <string>

#include "../util/axis_util.h"
#include "error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "op_log.h"
#include "register/op_impl_registry.h"

namespace {
const size_t kMatmulV2MinShapeSize = 2;
const size_t kMatmulV2MaxShapeSize = 4;
const size_t kBatchMatmulMinShapeSize = 2;
const size_t kBatchMatmulMaxShapeSize = 8;
const size_t kBatchMatMulBiasIdx = 2;
const int64_t kBlockSize = 16;
}  // namespace

namespace gert {
static void InferComplementedOutput(bool shape_x1_reshape_flag, bool shape_x2_reshape_flag, Shape &shape_out) {
  size_t dim_num = shape_out.GetDimNum();
  if (dim_num >= kBatchMatmulMinShapeSize) {
    if (shape_x1_reshape_flag && !shape_x2_reshape_flag) {
      shape_out.SetDim(dim_num - kBatchMatmulMinShapeSize, shape_out.GetDim(dim_num - 1));
      shape_out.SetDimNum(dim_num - 1);
    }

    if (!shape_x1_reshape_flag && shape_x2_reshape_flag) {
      shape_out.SetDimNum(dim_num - 1);
    }
  }
}

ge::graphStatus InferShapeForMatMul(InferShapeContext *context, bool is_matmul_v2) {
  auto op_name = context->GetNodeName();
  auto shape_x1 = context->GetInputShape(0);
  auto shape_x2 = context->GetInputShape(1);
  auto shape_out = context->GetOutputShape(0);
  auto tensor_x1 = context->GetInputDesc(0);
  auto attrs = context->GetAttrs();
  CHECK(shape_x1 == nullptr || shape_x2 == nullptr || shape_out == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "shape or attrs is null"), return ge::GRAPH_FAILED);

  const bool *trans_a = attrs->GetAttrPointer<bool>(0);
  const bool *trans_b = attrs->GetAttrPointer<bool>(1);
  CHECK(trans_a == nullptr || trans_b == nullptr, CUBE_INNER_ERR_REPORT(op_name, "attribute is null"),
        return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(), "x1_shape: %s, x2_shape: %s, transpose_x1: %d, transpose_x2: %d",
          ge::Shape2String(*shape_x1).c_str(), ge::Shape2String(*shape_x2).c_str(), *trans_a, *trans_b);

  ge::DataType dtype = tensor_x1->GetDataType();
  if (dtype == ge::DT_FLOAT) {
    OP_LOGW(op_name, "%s fp32 op has poor performance!", context->GetNodeName());
  }

  Shape shape_x1_new(*shape_x1);
  bool shape_x1_reshape_flag = false;
  if (shape_x1_new.GetDimNum() == 1 && shape_x1_new.GetDim(0) > 0) {
    shape_x1_reshape_flag = true;
    int64_t ori_dim = shape_x1_new.GetDim(0);
    shape_x1_new.SetDimNum(kBatchMatmulMinShapeSize);
    shape_x1_new.SetDim(0, 1);
    shape_x1_new.SetDim(1, ori_dim);
  }

  Shape shape_x2_new(*shape_x2);
  bool shape_x2_reshape_flag = false;
  if (shape_x2_new.GetDimNum() == 1 && shape_x2_new.GetDim(0) > 0) {
    shape_x2_reshape_flag = true;
    int64_t ori_dim = shape_x2_new.GetDim(0);
    shape_x2_new.SetDimNum(kBatchMatmulMinShapeSize);
    shape_x2_new.SetDim(0, ori_dim);
    shape_x2_new.SetDim(1, 1);
  }

  const Shape *shape_bias = nullptr;
  if (is_matmul_v2) {
    if (attrs->GetAttrNum() >= 5) {  // 5 attrs at least: transpose_x1, transpose_x2, offset_x, input_size, hidden_size
      auto input_size = attrs->GetAttrPointer<int64_t>(3);   // 3: input_size
      auto hidden_size = attrs->GetAttrPointer<int64_t>(4);  // 4: hidden_size
      if (input_size != nullptr && hidden_size != nullptr && *input_size > 0 && *hidden_size > 0) {
        OP_LOGD(op_name, "get private attr, input_size: %ld, hidden_size: %ld", *input_size, *hidden_size);
        shape_x2_new.SetDim(1, shape_x1_new.GetDim(1));
        int64_t align_dim = (*input_size + kBlockSize - 1) / kBlockSize * kBlockSize +
                            (*hidden_size + kBlockSize) / kBlockSize * kBlockSize;
        shape_x2_new.SetDim(0, align_dim);
      }
    }

    shape_bias = context->GetOptionalInputShape(kBatchMatMulBiasIdx);
    OP_LOGD(op_name, "check the input shape length.");
    if (shape_x1_new.GetDimNum() != kMatmulV2MinShapeSize && shape_x1_new.GetDimNum() != kMatmulV2MaxShapeSize) {
      CUBE_INNER_ERR_REPORT(op_name, "first input dim num[%zu] is not 2 or 4!", shape_x1_new.GetDimNum());
      return ge::GRAPH_FAILED;
    }
  }

  size_t idx_m = *trans_a ? 1 : 0;
  size_t idx_k_a = *trans_a ? 0 : 1;
  size_t idx_k_b = *trans_b ? 1 : 0;
  size_t idx_n = *trans_b ? 0 : 1;
  if (shape_x1_new.GetDim(idx_k_a) != shape_x2_new.GetDim(idx_k_b)) {
    CUBE_INNER_ERR_REPORT(op_name, "The k-axis of a(%ld) and b(%ld) tensors must be the same",
                          shape_x1_new.GetDim(idx_k_a), shape_x2_new.GetDim(idx_k_b));
    return ge::GRAPH_FAILED;
  }
  shape_out->SetDimNum(kBatchMatmulMinShapeSize);
  shape_out->SetDim(0, shape_x1_new.GetDim(idx_m));
  shape_out->SetDim(1, shape_x2_new.GetDim(idx_n));
  if (shape_bias != nullptr && shape_bias->GetDimNum() > 0) {
    int64_t bias_dim = shape_bias->GetDimNum();
    CHECK(shape_bias->GetDim(bias_dim - 1) != shape_out->GetDim(1),
          OP_LOGE(op_name, "The dimension of n [%ld] and bias [%ld] tensors must be the same", shape_out->GetDim(1),
                  shape_bias->GetDim(bias_dim - 1)),
          return ge::GRAPH_FAILED);
  }

  InferComplementedOutput(shape_x1_reshape_flag, shape_x2_reshape_flag, *shape_out);

  OP_LOGD(op_name, "end infershape.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForMatMul(InferShapeContext *context) {
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMul", "context is null"), return ge::GRAPH_FAILED);
  return InferShapeForMatMul(context, false);
}

ge::graphStatus InferShapeForMatMulV2(InferShapeContext *context) {
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMulV2", "context is null"), return ge::GRAPH_FAILED);
  return InferShapeForMatMul(context, true);
}

class InferShapeBatchMatMul {
 public:
  InferShapeBatchMatMul(InferShapeContext *context, const Shape &shape_a, const Shape &shape_b, bool trans_a,
                        bool trans_b)
      : op_name(context->GetNodeName()),
        shape_a(shape_a),
        shape_b(shape_b),
        trans_a(trans_a),
        trans_b(trans_b),
        shape_out(*(context->GetOutputShape(0))),
        shape_bias(context->GetOptionalInputShape(kBatchMatMulBiasIdx)) {
    num_dima = shape_a.GetDimNum();
    num_dimb = shape_b.GetDimNum();
    num_dim = std::max(num_dima, num_dimb);
    num_dim_bias = 0;
    if (shape_bias != nullptr) {
      num_dim_bias = shape_bias->GetDimNum();
      num_dim = std::max(num_dim, num_dim_bias);
    }
    shape_out.SetDimNum(num_dim);
  };

  ~InferShapeBatchMatMul(){};
  bool InferShape();

 protected:
  bool InferBatch() const;
  bool InferBias();

  size_t num_dim;
  size_t num_dima;
  size_t num_dimb;
  size_t num_dim_bias;

  const char *op_name;
  const Shape &shape_a;
  const Shape &shape_b;
  bool trans_a;
  bool trans_b;
  Shape &shape_out;
  const Shape *shape_bias;
};

void CopyOutShapeFromInputShape(const Shape &shape_in, Shape &shape_out, int64_t valid_offset) {
  for (auto i = 0; i < valid_offset; ++i) {
    shape_out.SetDim(i, shape_in.GetDim(i));
  }
}

bool InferShapeBatchMatMul::InferBatch() const {
  auto valid_offset = num_dim - std::min(num_dima, num_dimb);
  const Shape &shape_long = num_dima < num_dimb ? shape_b : shape_a;
  const Shape &shape_short = num_dima < num_dimb ? shape_a : shape_b;
  int64_t shape_value_long;
  int64_t shape_value_short;

  CopyOutShapeFromInputShape(shape_long, shape_out, valid_offset);
  // use index - 2 to get index of m
  for (auto i = valid_offset; i < num_dim - 2; ++i) {
    shape_value_short = shape_short.GetDim(i - valid_offset);
    shape_value_long = shape_long.GetDim(i);
    if (shape_value_short > 1 && shape_value_long > 1 && shape_value_short != shape_value_long) {
      return false;
    }
    shape_out.SetDim(i, std::max(shape_value_short, shape_value_long));
  }
  return true;
}

bool BroadcastBatchDim(const char *op_name, const int64_t dim_a, const int64_t dim_b, int64_t &dim) {
  if (dim_a > 1 && dim_b > 1) {
    CHECK(dim_a != dim_b,
          CUBE_INNER_ERR_REPORT(op_name, "[InferShape] dimensions a(%ld) and b(%ld) must be equal", dim_a, dim_b),
          return false);

    dim = dim_a;
    return true;
  }

  dim = std::max(dim_a, dim_b);
  return true;
}

bool InferNDimWithBias(const char *op_name, const int64_t dim_a, const int64_t dim_b, int64_t &dim) {
  // shape_bias_n > 0 && n > 0
  if (dim_a > 0 && dim_b > 0) {
    CHECK(dim_a != dim_b,
          CUBE_INNER_ERR_REPORT(op_name, "[InferShape] dimensions a(%ld) and b(%ld) must be equal", dim_a, dim_b),
          return false);
    dim = dim_a;
    return true;
  }

  return false;
}

bool InferShapeBatchMatMul::InferBias() {
  int64_t shape_value_out = shape_out.GetDim(num_dim - 1);
  // 1) shape_bias = {}
  CHECK(num_dim_bias == 0, CUBE_INNER_ERR_REPORT(op_name, "[InferShape] bias dims number is zero"), return true);

  // 2) infer n with bias
  CHECK(
      !InferNDimWithBias(op_name, shape_bias->GetDim(num_dim_bias - 1), shape_out.GetDim(num_dim - 1), shape_value_out),
      CUBE_INNER_ERR_REPORT(op_name, "[InferShape] failed to infer N dim with bias"), return false);

  shape_out.SetDim(num_dim - 1, shape_value_out);

  // 3) infer batch with bias
  auto valid_offset = num_dim - std::min(num_dim_bias, std::max(num_dima, num_dimb));
  if (num_dim_bias < num_dim) {
    // stop before num_dim - 2 so as to avoid traversing axis m, n
    for (auto i = valid_offset; i < num_dim - 2; ++i) {
      CHECK(!BroadcastBatchDim(op_name, shape_bias->GetDim(i - valid_offset), shape_out.GetDim(i), shape_value_out),
            CUBE_INNER_ERR_REPORT(op_name, "[InferShape] failed to broadcast batch dim"), return false);

      shape_out.SetDim(i, shape_value_out);
    }
    return true;
  }
  CopyOutShapeFromInputShape(*shape_bias, shape_out, valid_offset);
  // stop before num_dim - 2 so as to avoid traversing axis m, n
  for (auto i = valid_offset; i < num_dim - 2; ++i) {
    CHECK(!BroadcastBatchDim(op_name, shape_bias->GetDim(i), shape_out.GetDim(i - valid_offset), shape_value_out),
          CUBE_INNER_ERR_REPORT(op_name, "[InferShape] failed to broadcast batch dim"), return false);

    shape_out.SetDim(i, shape_value_out);
  }
  return true;
}

bool InferShapeBatchMatMul::InferShape() {
  if (shape_a.GetDimNum() < kBatchMatmulMinShapeSize || shape_b.GetDimNum() < kBatchMatmulMinShapeSize) {
    CHECK(!InferBatch(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed to x1/x2 dim num less than 2."),
          return false);
    return false;
  }
  // using index - 2 to get m_dim
  size_t idx_m = num_dima - 2;
  size_t idx_k_a = num_dima - 1;
  // using index - 2 to get k_dim
  size_t idx_k_b = num_dimb - 2;
  size_t idx_n = num_dimb - 1;
  if (trans_a) {
    idx_m = num_dima - 1;
    // using index - 2 to get k_dim
    idx_k_a = num_dima - 2;
  }
  if (trans_b) {
    idx_k_b = num_dimb - 1;
    // using index - 2 to get n_dim
    idx_n = num_dimb - 2;
  }

  if (shape_a.GetDim(idx_k_a) != shape_b.GetDim(idx_k_b)) {
    CUBE_INNER_ERR_REPORT(op_name, "[InferShape] The k-axis of a(%ld) and b(%ld) tensors must be the same",
                          shape_a.GetDim(idx_k_a), shape_b.GetDim(idx_k_b));
    return false;
  }
  CHECK(!InferBatch(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed to infer Batch."), return false);

  // using index - 2 to get m_dim in shape_out
  shape_out.SetDim((num_dim - 2), shape_a.GetDim(idx_m));
  shape_out.SetDim((num_dim - 1), shape_b.GetDim(idx_n));
  if (shape_bias != nullptr) {
    CHECK(!InferBias(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Infer bias failed."), return false);
  }

  return true;
}

ge::graphStatus InferShapeForBatchMatMulV2(InferShapeContext *context) {
  CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("BatchMatMulV2", "context is null"), return ge::GRAPH_FAILED);

  auto shape_x1 = context->GetInputShape(0);
  auto shape_x2 = context->GetInputShape(1);
  auto shape_out = context->GetOutputShape(0);
  auto attrs = context->GetAttrs();
  auto op_name = context->GetNodeName();
  CHECK(shape_x1 == nullptr || shape_x2 == nullptr || shape_out == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "[Infershape]shape is null"), return ge::GRAPH_FAILED);

  const bool *adj_x1 = attrs->GetAttrPointer<bool>(0);
  const bool *adj_x2 = attrs->GetAttrPointer<bool>(1);
  CHECK(adj_x1 == nullptr || adj_x2 == nullptr, CUBE_INNER_ERR_REPORT(op_name, "[Infershape]attribute is null"),
        return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(), "x1_shape: %s, x2_shape: %s, adj_x1: %d, adj_x2: %d",
          ge::Shape2String(*shape_x1).c_str(), ge::Shape2String(*shape_x2).c_str(), *adj_x1, *adj_x2);

  auto dim_num = std::max(shape_x1->GetDimNum(), shape_x2->GetDimNum());
  if (dim_num < 1 || dim_num > kBatchMatmulMaxShapeSize) {
    CUBE_INNER_ERR_REPORT(op_name, "[Infershape]The shape can only be in the range of 1 to 8.");
  }

  Shape shape_x2_new(*shape_x2);
  bool shape_x2_reshape_flag = false;
  if (shape_x2_new.GetDimNum() == 1 && shape_x2_new.GetDim(0) > 0) {
    shape_x2_reshape_flag = true;
    int64_t ori_dim = shape_x2_new.GetDim(0);
    shape_x2_new.SetDimNum(kBatchMatmulMinShapeSize);
    shape_x2_new.SetDim(0, ori_dim);
    shape_x2_new.SetDim(1, 1);
  }

  Shape shape_x1_new(*shape_x1);
  bool shape_x1_reshape_flag = false;
  if (shape_x1_new.GetDimNum() == 1 && shape_x1_new.GetDim(0) > 0) {
    shape_x1_reshape_flag = true;
    int64_t ori_dim = shape_x1_new.GetDim(0);
    shape_x1_new.SetDimNum(kBatchMatmulMinShapeSize);
    shape_x1_new.SetDim(0, 1);
    shape_x1_new.SetDim(1, ori_dim);
  }

  InferShapeBatchMatMul BatchMatMulV2Infer(context, shape_x1_new, shape_x2_new, *adj_x1, *adj_x2);
  CHECK(!BatchMatMulV2Infer.InferShape(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed to infer output shape"),
        return ge::GRAPH_FAILED);

  InferComplementedOutput(shape_x1_reshape_flag, shape_x2_reshape_flag, *shape_out);
  OP_LOGD(context->GetNodeName(), "output shape: %s", ge::Shape2String(*(context->GetOutputShape(0))).c_str());
  // no need to SetDataType in runtime
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(BatchMatMul).InferShape(InferShapeForBatchMatMulV2);

IMPL_OP(BatchMatMulV2).InferShape(InferShapeForBatchMatMulV2);

IMPL_OP(MatMul).InferShape(InferShapeForMatMul);

IMPL_OP(MatMulV2).InferShape(InferShapeForMatMulV2);
}  // namespace gert
