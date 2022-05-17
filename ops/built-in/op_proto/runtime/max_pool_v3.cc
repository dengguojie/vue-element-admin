/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "runtime_util.h"

using namespace ge;
namespace ops {
constexpr size_t INDEX_KSIZE = 0;
constexpr size_t INDEX_STRIDES = 1;
constexpr size_t INDEX_PADDING_MODE = 2;
constexpr size_t INDEX_PADS = 3;
constexpr size_t INDEX_DATA_FORMAT = 4;
constexpr size_t INDEX_GLOBAL_POOLING = 5;
constexpr size_t INDEX_CEIL_MODE = 6;
constexpr size_t SHAPE_NHWC_SIZE = 4;
constexpr size_t PAD_SIZE = 4;
constexpr size_t PAD_TOP = 0;
constexpr size_t PAD_BOTTOM = 1;
constexpr size_t PAD_LEFT = 2;
constexpr size_t PAD_RIGHT = 3;

static int64_t SameUpdateDim(const int64_t ksize, const int64_t strides, int64_t dim_size) {
  CHECK_DIVISOR_ZERO_RET(strides, dim_size - ksize + 1);
  return (dim_size - ksize + strides) / strides;
}

static void CalculateUpdateDim(const int64_t ksize, const int64_t strides, const bool ceil_mode,
                               const int64_t pad_a, const int64_t pad_b, int64_t& dim_size) {
  CHECK_DIVISOR_ZERO(strides);
  if (ceil_mode) {
    dim_size = (dim_size + pad_a + pad_b - ksize + strides + strides - 1) / strides;
  } else {
    dim_size = (dim_size + pad_a + pad_b - ksize + strides) / strides;
  }
}

ge::graphStatus SameUpdateHWDim(gert::InferShapeContext *context, size_t h_dim,
                                size_t w_dim, const int64_t *strides_data,
                                const gert::Shape *in_shape, gert::Shape *out_shape) {
  int64_t dim_size = in_shape->GetDim(h_dim);
  out_shape->SetDim(h_dim, SameUpdateDim(1, strides_data[h_dim], dim_size));
  dim_size = in_shape->GetDim(w_dim);
  out_shape->SetDim(w_dim, SameUpdateDim(1, strides_data[w_dim], dim_size));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalculateUpdateHWDim(gert::InferShapeContext *context, size_t h_dim, size_t w_dim,
                                     const gert::RuntimeAttrs *attrs, const int64_t *strides_data,
                                     const gert::Shape *in_shape, gert::Shape *out_shape) {
  auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
  OPS_CHECK_NULL_WITH_CONTEXT(context, ksize);
  OP_CHECK(ksize->GetSize() != SHAPE_NHWC_SIZE,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Length of ksize must be 4!"),
           return GRAPH_FAILED);
  auto ksize_data = reinterpret_cast<const int64_t *>(ksize->GetData());
  auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PADS);
  OPS_CHECK_NULL_WITH_CONTEXT(context, pads);
  OP_CHECK(pads->GetSize() != PAD_SIZE,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Length of pads must be 4!"),
           return GRAPH_FAILED);
  auto ceil_mode = attrs->GetAttrPointer<bool>(INDEX_CEIL_MODE);
  OPS_CHECK_NULL_WITH_CONTEXT(context, ceil_mode);
  auto pads_data = reinterpret_cast<const int64_t *>(pads->GetData());
  int64_t dim_size = in_shape->GetDim(h_dim);
  CalculateUpdateDim(ksize_data[h_dim], strides_data[h_dim], *ceil_mode,
                     pads_data[PAD_TOP], pads_data[PAD_BOTTOM], dim_size);
  out_shape->SetDim(h_dim,dim_size);
  dim_size = in_shape->GetDim(w_dim);
  CalculateUpdateDim(ksize_data[w_dim], strides_data[w_dim], *ceil_mode,
                     pads_data[PAD_LEFT], pads_data[PAD_RIGHT], dim_size);
  out_shape->SetDim(w_dim, dim_size);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForMaxPoolV3(gert::InferShapeContext *context) {
  auto in_shape = context->GetInputShape(0);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  *out_shape = *in_shape;

  auto src_td = context->GetInputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, src_td);
  auto input_format = src_td->GetStorageFormat();
  size_t h_dim = input_format == FORMAT_NHWC ? 1 : 2;
  size_t w_dim = input_format == FORMAT_NHWC ? 2 : 3;

  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const bool *global_pooling = attrs->GetAttrPointer<bool>(INDEX_GLOBAL_POOLING);
  OPS_CHECK_NULL_WITH_CONTEXT(context, global_pooling);
  if (*global_pooling) {
    out_shape->SetDim(h_dim, 1);
    out_shape->SetDim(w_dim, 1);
  } else {
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK(strides->GetSize() != SHAPE_NHWC_SIZE,
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Length of strides must be 4!"),
             return GRAPH_FAILED);
    auto strides_data = reinterpret_cast<const int64_t *>(strides->GetData());
    OP_CHECK(strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0,
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "strides h and w must be greater than 0."),
             return GRAPH_FAILED);
    auto padding_mode = attrs->GetAttrPointer<char>(INDEX_PADDING_MODE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, padding_mode);
    if (strcmp(padding_mode, "CALCULATED") == 0) {
      // when padding_mode is "CALCULATED"
      return CalculateUpdateHWDim(context, h_dim, w_dim, attrs, strides_data, in_shape, out_shape);
    } else {
      // when padding_mode in ("SAME", "VALID")
      return SameUpdateHWDim(context, h_dim, w_dim, strides_data, in_shape, out_shape);
    }
  }

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(MaxPoolV3)
    .InferShape(InferShapeForMaxPoolV3);
}  // namespace ops
