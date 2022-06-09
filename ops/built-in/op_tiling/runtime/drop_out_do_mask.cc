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

#include "drop_out_do_mask.h"

#include "op_tiling_util.h"
#include "runtime2_util.h"

namespace optiling {
const int64_t CORE_MINEST_NUM = 128;

static void SetRuningParams(DropOutDoMaskTilingData* params, const int64_t core_used_num, const int64_t num_per_core,
                            const int64_t num_tail_core) {
  params->core_used_num = core_used_num;
  params->num_per_core = num_per_core;
  params->num_tail_core = num_tail_core;
}

ge::graphStatus DropOutDoMaskTiling(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "DropOutDoMaskTiling running begin");
  auto compile_info = reinterpret_cast<const DropOutDoMaskCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  auto params = context->GetTilingData<DropOutDoMaskTilingData>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, params);

  auto var_desc = context->GetInputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, var_desc);
  auto& var_shape = context->GetInputShape(0)->GetStorageShape();
  int64_t dim_num = var_shape.GetDimNum();
  int64_t var_num = dim_num == 0 ? 1 : GetPartShapeSize(var_shape, 0, dim_num);
  OP_LOGD(context->GetNodeName(), "var_num is %ld", var_num);
  int64_t core_num = compile_info->core_num;
  OP_LOGD(context->GetNodeName(), "core_num is %ld", core_num);

  int64_t sigment_total = (var_num + CORE_MINEST_NUM - 1) / CORE_MINEST_NUM;
  int64_t sigment_per_core = (sigment_total + core_num - 1) / core_num;
  int64_t core_used_num = sigment_per_core == 0 ? 1 : (sigment_total + sigment_per_core - 1) / sigment_per_core;
  int64_t num_per_core = sigment_per_core * CORE_MINEST_NUM;
  int64_t num_tail_core = var_num - (num_per_core * (core_used_num - 1));
  OP_LOGD(context->GetNodeName(), "CompileParams, core_used_num = %d", core_used_num);
  OP_LOGD(context->GetNodeName(), "CompileParams, num_per_core = %d", num_per_core);
  OP_LOGD(context->GetNodeName(), "CompileParams, num_tail_core = %d", num_tail_core);
  SetRuningParams(params, core_used_num, num_per_core, num_tail_core);
  context->SetBlockDim(core_num);
  OP_LOGD(context->GetNodeName(), "DropOutDoMaskTiling run success.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDropOutDoMask(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "TilingPrepareForDropOutDoMask running.");
  auto compile_info = MutableCompileInfo<DropOutDoMaskCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  const nlohmann::json& all_vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(all_vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get vars failed."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(
      !GetCompileValue(all_vars, "core_num", compile_info->core_num),
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "TilingPrepareForDropOutDoMask, get core_num error"),
      return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(), "TilingPrepareForDropOutDoMask GRAPH_SUCCESS.");
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the DropOutDoMask op.
IMPL_OP(DropOutDoMask).Tiling(DropOutDoMaskTiling).TilingParse<DropOutDoMaskCompileInfo>(TilingPrepareForDropOutDoMask);
}  // namespace optiling
