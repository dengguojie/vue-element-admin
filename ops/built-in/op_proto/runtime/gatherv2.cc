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
#include "op_const.h"

using namespace ge;
namespace ops {
const size_t INPUT_IDX_X = 0;
const size_t INPUT_IDX_INDICES = 1;
const size_t INPUT_IDX_AXIS = 2;

bool CheckAndUpdateAxis(gert::InferShapeContext* context, int64_t& batch_dims, int64_t& axes_data,
                        int64_t rank_indices, int64_t x_real_dim_cnt) {
  OP_CHECK(batch_dims < -rank_indices || (batch_dims >= rank_indices && rank_indices != 0),
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "batch_dims is invalid"), return false);
  if (batch_dims < 0) {
    batch_dims = batch_dims + rank_indices;
  }
  OP_CHECK(batch_dims >= x_real_dim_cnt,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "batch_dims must be less than rank x)"),
           return false);
  auto x_shape = context->GetInputShape(0);
  auto indies_shape = context->GetInputShape(1);
  for (int i = 0; i < batch_dims; i++) {
    if (x_shape->GetDim(i) != indies_shape->GetDim(i) && x_shape->GetDim(i) > 0 && indies_shape->GetDim(i) > 0) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "x_shape not equal indies_shape");
      return false;
    }
  }
  OP_CHECK(axes_data > 0 && (x_real_dim_cnt < axes_data + 1),
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "axis is invalid"), return false);
  if (axes_data < 0) {
    OP_CHECK(x_real_dim_cnt < -axes_data,
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "axis is invalid"), return false);

    axes_data = x_real_dim_cnt + axes_data;
    OP_CHECK(
        batch_dims > axes_data,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "batch_dims must be less than or equal to axis"),
        return false);
  }
  return true;
}

ge::graphStatus GatgerV2Impl(gert::InferShapeContext* context, int64_t& axes_data, const gert::Shape* x_shape,
                             const gert::Shape* indies_shape, gert::Shape* out_shape) {
  OP_LOGD("GatgerV2Impl", "gatherv2 infershape impl is begin");
  int64_t x_real_dim_cnt = x_shape->GetDimNum();
  int64_t rank_indices = indies_shape->GetDimNum();
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const auto* batchdims = attrs->GetAttrPointer<int64_t>(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, batchdims);
  int64_t batch_dims = *batchdims;

  OP_CHECK(x_real_dim_cnt < 1,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT("x_real_dim_cnt", "x_real_dim_cnt must be more than 1"),
           return ge::GRAPH_FAILED);
  OP_CHECK(!CheckAndUpdateAxis(context, batch_dims, axes_data, rank_indices, x_real_dim_cnt),
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "check axis and batchdim failed"),
           return ge::GRAPH_FAILED);

  for (int64_t i = 0; i < axes_data; i++) {
    out_shape->AppendDim(x_shape->GetDim(i));
  }
  // real dim cnt has no existing meaning .Original shape has replace its meaning now
  for (int64_t i = batch_dims; i < rank_indices; i++) {
    out_shape->AppendDim(indies_shape->GetDim(i));
  }

  for (int64_t i = axes_data + 1; i < x_real_dim_cnt; i++) {
    out_shape->AppendDim(x_shape->GetDim(i));
  }
  return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForGatherV2(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "infershape is begin");
  auto x_shape = context->GetInputShape(INPUT_IDX_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  auto indies_shape = context->GetInputShape(INPUT_IDX_INDICES);
  OPS_CHECK_NULL_WITH_CONTEXT(context, indies_shape);
  auto axes_tensor = context->GetInputTensor(INPUT_IDX_AXIS);
  OPS_CHECK_NULL_WITH_CONTEXT(context, axes_tensor);
  auto out_shape = context->GetOutputShape(INPUT_IDX_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  auto axes_size = static_cast<int32_t>(axes_tensor->GetShapeSize());
  OP_LOGD(context->GetNodeName(), "axes_size is %d", axes_size);
  OP_CHECK(axes_size < 1, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "axes size must big than 0!"),
           return ge::GRAPH_FAILED);
  int64_t axis = 0;
  OP_CHECK(!GetConstInt(context, INPUT_IDX_AXIS, axis),
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "get axis failed"), return ge::GRAPH_FAILED);

  return GatgerV2Impl(context, axis, x_shape, indies_shape, out_shape);
  OP_LOGD(context->GetNodeName(), "infershape is success");
}

IMPL_OP(GatherV2).InferShape(InferShapeForGatherV2).InputsDataDependency({INPUT_IDX_AXIS});
}  // namespace ops
