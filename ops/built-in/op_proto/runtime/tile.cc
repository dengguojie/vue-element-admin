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
#include "op_util.h"

using namespace ge;
namespace ops {
constexpr size_t INDEX_MULTIPLES = 0;
constexpr size_t TILE_IN_IDX = 0;
constexpr size_t TILE_OUT_IDX = 0;
static constexpr size_t MAXDIMNUM = 8;

ge::graphStatus TileInferShapeCommon(gert::InferShapeContext* context, const int64_t* multiples_data,
                                     size_t multiples_len) {
  auto in_shape = context->GetInputShape(TILE_IN_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto out_shape = context->GetOutputShape(TILE_OUT_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  auto in_shape_len = in_shape->GetDimNum();
  OP_CHECK(multiples_len < in_shape_len,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                               OtherErrMsg("the tile multiples len is less than the input len")),
           return ge::GRAPH_FAILED);
  OP_CHECK(multiples_len > MAXDIMNUM,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                               OtherErrMsg("the tile multiples len is more than MaxDimNum 8")),
           return ge::GRAPH_FAILED);
  // align shape for input
  gert::Shape in_shape_new;
  if (in_shape_len != multiples_len) {
    OP_LOGI(context->GetNodeName(), "align input shape with multiples.");
    int32_t len_diff = multiples_len - in_shape_len;
    for (int32_t i = 0; i < len_diff; i++) {
      in_shape_new.AppendDim(1);
    }
    for (size_t i = 0; i < in_shape_len; i++) {
      in_shape_new.AppendDim(in_shape->GetDim(i));
    }
    in_shape_len = multiples_len;
  } else {
    in_shape_new = *in_shape;
  }
  // in shape == [], out shape = []
  if (in_shape_len == 0) {
    OP_LOGI(context->GetNodeName(), "input shape is [], output shape is [].");
    *out_shape = *in_shape;
    return GRAPH_SUCCESS;
  }
  // calculate output shape dim value
  out_shape->SetDimNum(in_shape_len);
  for (uint64_t i = 0; i < in_shape_len; i++) {
    if (in_shape_new[i] >= 0) {
      out_shape->SetDim(i, in_shape_new[i] * multiples_data[i]);
    } else {
      std::string err_msg = ConcatString("Runtime infershape illegal input dim:", i, ", value is ", in_shape_new[i]);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), OtherErrMsg(err_msg));
      return ge::GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForTileD(gert::InferShapeContext* context) {
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  auto multiples = attrs->GetListInt(INDEX_MULTIPLES);
  OPS_CHECK_NULL_WITH_CONTEXT(context, multiples);
  const int64_t* multiples_data = multiples->GetData();
  size_t multiples_len = multiples->GetSize();
  return TileInferShapeCommon(context, multiples_data, multiples_len);
}

IMPL_OP(TileD).InferShape(InferShapeForTileD);
}  // namespace ops
