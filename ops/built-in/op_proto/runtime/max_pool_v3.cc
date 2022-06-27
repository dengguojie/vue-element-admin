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

typedef ge::graphStatus (*InferShapePaddingFunc)(gert::InferShapeContext*, size_t, size_t, const gert::RuntimeAttrs*);

static std::string Int64ToString(const int64_t* data, size_t size) {
  std::string r = "[";
  for (size_t i = 0; i < size; i++) {
    r = r + std::to_string(data[i]) + " ";
  }
  r = r + "]";
  return r;
}

static int64_t SameUpdateDim(const int64_t ksize, const int64_t strides, int64_t dim_size) {
  return (strides == 0) ? (dim_size - ksize + 1) : ((dim_size - ksize + strides) / strides);
}

static void CalculateUpdateDim(const int64_t ksize, const int64_t strides, const bool ceil_mode, int64_t& dim_size) {
  if (ceil_mode) {
    dim_size = (strides == 0) ? (dim_size - ksize + 1) : (dim_size - ksize + strides + strides - 1) / strides;
  } else {
    dim_size = (strides == 0) ? (dim_size - ksize + 1) : (dim_size - ksize + strides) / strides;
  }
}

ge::graphStatus InferShapeGlobalPooling(gert::InferShapeContext* context, size_t h_dim, size_t w_dim) {
  auto in_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  *out_shape = *in_shape;
  out_shape->SetDim(h_dim, 1);
  out_shape->SetDim(w_dim, 1);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapePaddingCalculated(gert::InferShapeContext* context, size_t h_dim, size_t w_dim,
                                            const gert::RuntimeAttrs *attrs) {
  auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
  OPS_CHECK_NULL_WITH_CONTEXT(context, ksize);
  OP_CHECK(ksize->GetSize() != SHAPE_NHWC_SIZE,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context->GetNodeName(), ConcatString("Length of ksize ", ksize->GetSize(), " must be 4!")),
           return GRAPH_FAILED);
  auto ksize_data = reinterpret_cast<const int64_t*>(ksize->GetData());
  auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PADS);
  OPS_CHECK_NULL_WITH_CONTEXT(context, pads);
  OP_CHECK(pads->GetSize() != PAD_SIZE,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context->GetNodeName(), ConcatString("Length of pads ", pads->GetSize(), " must be 4!")),
           return GRAPH_FAILED);
  auto ceil_mode = attrs->GetAttrPointer<bool>(INDEX_CEIL_MODE);
  OPS_CHECK_NULL_WITH_CONTEXT(context, ceil_mode);
  auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
  OPS_CHECK_NULL_WITH_CONTEXT(context, strides);
  OP_CHECK(strides->GetSize() != SHAPE_NHWC_SIZE,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context->GetNodeName(), ConcatString("Length of strides ", strides->GetSize(), " must be 4!")),
           return GRAPH_FAILED);
  auto strides_data = reinterpret_cast<const int64_t*>(strides->GetData());
  OP_CHECK(strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context->GetNodeName(),
               ConcatString(Int64ToString(strides_data, strides->GetSize()), " h ", strides_data[h_dim],
                            " and w ", strides_data[w_dim], " must be greater than 0.")),
           return GRAPH_FAILED);
  auto pads_data = reinterpret_cast<const int64_t*>(pads->GetData());

  auto in_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  *out_shape = *in_shape;
  int64_t dim_size = in_shape->GetDim(h_dim);
  dim_size += pads_data[PAD_TOP] + pads_data[PAD_BOTTOM];
  CalculateUpdateDim(ksize_data[h_dim], strides_data[h_dim], *ceil_mode, dim_size);
  out_shape->SetDim(h_dim, dim_size);
  dim_size = in_shape->GetDim(w_dim);
  dim_size += pads_data[PAD_LEFT] + pads_data[PAD_RIGHT];
  CalculateUpdateDim(ksize_data[w_dim], strides_data[w_dim], *ceil_mode, dim_size);
  out_shape->SetDim(w_dim, dim_size);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapePaddingValid(gert::InferShapeContext* context, size_t h_dim, size_t w_dim,
                                       const gert::RuntimeAttrs *attrs) {
  auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
  OPS_CHECK_NULL_WITH_CONTEXT(context, ksize);
  OP_CHECK(ksize->GetSize() != SHAPE_NHWC_SIZE,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context->GetNodeName(), ConcatString("Length of ksize ", ksize->GetSize(), " must be 4!")),
           return GRAPH_FAILED);
  auto ksize_data = reinterpret_cast<const int64_t*>(ksize->GetData());
  auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
  OPS_CHECK_NULL_WITH_CONTEXT(context, strides);
  OP_CHECK(strides->GetSize() != SHAPE_NHWC_SIZE,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context->GetNodeName(), ConcatString("Length of strides ", strides->GetSize(), " must be 4!")),
           return GRAPH_FAILED);
  auto strides_data = reinterpret_cast<const int64_t*>(strides->GetData());
  OP_CHECK(strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context->GetNodeName(),
               ConcatString(Int64ToString(strides_data, strides->GetSize()), " h ", strides_data[h_dim],
                            " and w ", strides_data[w_dim], " must be greater than 0.")),
           return GRAPH_FAILED);

  auto in_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  *out_shape = *in_shape;

  int64_t dim_size = in_shape->GetDim(h_dim);
  out_shape->SetDim(h_dim, SameUpdateDim(ksize_data[h_dim], strides_data[h_dim], dim_size));
  dim_size = in_shape->GetDim(w_dim);
  out_shape->SetDim(w_dim, SameUpdateDim(ksize_data[w_dim], strides_data[w_dim], dim_size));

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapePaddingSame(gert::InferShapeContext* context, size_t h_dim, size_t w_dim,
                                      const gert::RuntimeAttrs *attrs) {
  auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
  OPS_CHECK_NULL_WITH_CONTEXT(context, strides);
  OP_CHECK(strides->GetSize() != SHAPE_NHWC_SIZE,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context->GetNodeName(), ConcatString("Length of strides ", strides->GetSize(), " must be 4!")),
           return GRAPH_FAILED);
  auto strides_data = reinterpret_cast<const int64_t*>(strides->GetData());
  OP_CHECK(strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context->GetNodeName(),
               ConcatString(Int64ToString(strides_data, strides->GetSize()), " h ", strides_data[h_dim],
                            " and w ", strides_data[w_dim], " must be greater than 0.")),
           return GRAPH_FAILED);

  auto in_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  *out_shape = *in_shape;

  int64_t dim_size = in_shape->GetDim(h_dim);
  out_shape->SetDim(h_dim, SameUpdateDim(1, strides_data[h_dim], dim_size));
  dim_size = in_shape->GetDim(w_dim);
  out_shape->SetDim(w_dim, SameUpdateDim(1, strides_data[w_dim], dim_size));

  return ge::GRAPH_SUCCESS;
}

static const std::vector<std::pair<std::string, InferShapePaddingFunc>> kFuncMap = {
    {"CALCULATED", InferShapePaddingCalculated},
    {"SAME", InferShapePaddingSame},
    {"VALID", InferShapePaddingValid},
};

ge::graphStatus InferShapeForMaxPoolV3(gert::InferShapeContext* context) {
  auto src_td = context->GetInputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, src_td);
  auto input_format = src_td->GetStorageFormat();
  size_t h_dim = input_format == FORMAT_NHWC ? 1 : 2;
  size_t w_dim = input_format == FORMAT_NHWC ? 2 : 3;

  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const bool* global_pooling = attrs->GetAttrPointer<bool>(INDEX_GLOBAL_POOLING);
  OPS_CHECK_NULL_WITH_CONTEXT(context, global_pooling);
  if (*global_pooling) {
    return InferShapeGlobalPooling(context, h_dim, w_dim);
  }

  auto padding_mode = attrs->GetAttrPointer<char>(INDEX_PADDING_MODE);
  OPS_CHECK_NULL_WITH_CONTEXT(context, padding_mode);
  auto it = std::find_if(
      kFuncMap.begin(), kFuncMap.end(), 
      [&padding_mode](const std::pair<std::string, InferShapePaddingFunc>& item)->bool{
        return item.first == padding_mode;
      });
  OP_CHECK(it == kFuncMap.end(),
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context->GetNodeName(),
               ConcatString("padding_mode ", padding_mode, " must in (CALCULATED, VALID, SAME).")),
           return GRAPH_FAILED);

  // when padding_mode in (CALCULATED, VALID, SAME)
  return it->second(context, h_dim, w_dim, attrs);
}

IMPL_OP(MaxPoolV3).InferShape(InferShapeForMaxPoolV3);
}  // namespace ops
