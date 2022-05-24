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
#include "flatten.h"
#include "runtime2_util.h"
#include "op_tiling_util.h"
#include "op_util.h"

using namespace ge;

namespace optiling {
ge::graphStatus TilingForFlatten(gert::TilingContext* context) {
  auto compile_info = reinterpret_cast<const FlattenCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  auto params = context->GetTilingData<FlattenTilingData>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, params);

  auto src_storage_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, src_storage_shape);
  auto src_shape = src_storage_shape->GetOriginShape();

  int64_t data_size = src_shape.GetDimNum() == 0 ? 1 : src_shape.GetShapeSize();

  int64_t core_num = compile_info->core_num;
  int64_t ub_size = compile_info->ub_size;
  int64_t block_size = compile_info->block_size;

  int64_t core_number = core_num;
  if (data_size < block_size) {
    core_number = 1;
  }
  params->core_data = CeilDiv(data_size, core_number);
  params->core_data = CeilDiv(params->core_data, block_size) * block_size;
  params->core_used = CeilDiv(data_size, params->core_data);
  int64_t core_last = params->core_data;
  if (data_size % params->core_data != 0) {
    core_last = data_size % params->core_data;
  }

  params->copy_loop = params->core_data / ub_size;
  params->copy_tail = params->core_data % ub_size;
  params->last_copy_loop = core_last / ub_size;
  params->last_copy_tail = core_last % ub_size;

  OP_LOGD("Flatten",
          "CompileParams, core_data = %d, core_used = %d, copy_loop = %d, copy_tail = %d, last_copy_loop = %d, "
          "last_copy_tail = %d",
          params->core_data, params->core_used, params->copy_loop, params->copy_tail, params->last_copy_loop,
          params->last_copy_tail);

  context->SetBlockDim(params->core_used);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForFlatten(gert::TilingParseContext* context) {
  auto compile_info = MutableCompileInfo<FlattenCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get vars failed."),
                  return ge::GRAPH_FAILED);
  GetCompileValue(vars, "ub_size", compile_info->ub_size);
  GetCompileValue(vars, "core_num", compile_info->core_num);
  GetCompileValue(vars, "block_size", compile_info->block_size);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Flatten).Tiling(TilingForFlatten).TilingParse<FlattenCompileInfo>(TilingPrepareForFlatten);
}  // namespace optiling
