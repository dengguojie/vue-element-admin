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
#include "dynamic_atomic_addr_clean.h"
#include "runtime2_util.h"
#include "op_tiling_util.h"
#include "op_util.h"

using namespace ge;

namespace optiling {
constexpr uint32_t MIN_ELE_SIZE_USING_ALL_CORE = 1024;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_FP32 = 4;
constexpr uint32_t MASK_FP32 = 64;
constexpr uint32_t MAX_REPEAT_TIME = 255;

static void ComputeParamsOneCore(const int32_t& ele_num_one_core,
                                 const int32_t& ele_num_full_mask_full_repeat_time_input_scalar,
                                 InputScalar& input_scalar) {
  int32_t scaler = ele_num_full_mask_full_repeat_time_input_scalar;
  scaler = scaler <= 0 ? MASK_FP32 * MAX_REPEAT_TIME : scaler;
  input_scalar.init_times_full_mask_full_repeat_time = ele_num_one_core / scaler;
  input_scalar.ele_num_front_part = input_scalar.init_times_full_mask_full_repeat_time * scaler;
  uint32_t ele_num_last_part = ele_num_one_core - input_scalar.ele_num_front_part;
  input_scalar.burst_len_last_part = CeilDiv(ele_num_last_part * BYTE_FP32, BYTE_BLOCK);
  if (ele_num_last_part % MASK_FP32 == 0) {
    input_scalar.repeat_time_last_part = ele_num_last_part / MASK_FP32;
  } else {
    input_scalar.repeat_time_last_part = ele_num_last_part / MASK_FP32 + 1;
  }
}

void InitTilingParams(DynamicAtomicAddrCleanTilingData* params) {
  std::memset(params, 0, sizeof(DynamicAtomicAddrCleanTilingData));
}

ge::graphStatus TilingForDynamicAtomicAddrClean(gert::TilingContext* context) {
  auto compile_info = reinterpret_cast<const DynamicAtomicAddrCleanCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  auto params = context->GetTilingData<DynamicAtomicAddrCleanTilingData>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, params);

  uint32_t core_num = compile_info->core_num;
  for (auto workspace_size : compile_info->_workspace_size_list) {
    OP_TILING_CHECK(
        ((workspace_size < 0) || (workspace_size % 32 != 0)),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                        "workspace size must be a natural multiple of 32, but it is %ld",
                                        workspace_size),
        return ge::GRAPH_FAILED);
    uint32_t ele_num_fp32 = workspace_size / BYTE_FP32;
    // init tiling params
    InitTilingParams(params);
    params->select_key_input_scalar = 1;
    // is using all core
    if (workspace_size >= MIN_ELE_SIZE_USING_ALL_CORE) {
      params->need_core_num_input_scalar = core_num;
    } else {
      params->need_core_num_input_scalar = 1;
    }
    // compute tiling params
    params->ele_num_full_mask_full_repeat_time_input_scalar = MASK_FP32 * MAX_REPEAT_TIME;
    params->burst_len_full_mask_full_repeat_time_input_scalar =
        params->ele_num_full_mask_full_repeat_time_input_scalar * BYTE_FP32 / BYTE_BLOCK;
    if (params->need_core_num_input_scalar == 1) {
      // use one core
      params->ele_num_front_core_input_scalar = ele_num_fp32;
      ComputeParamsOneCore(params->ele_num_front_core_input_scalar,
                           params->ele_num_full_mask_full_repeat_time_input_scalar, params->front_core_input_scalar);

      params->ele_num_last_core_input_scalar = params->ele_num_front_core_input_scalar;
      ComputeParamsOneCore(params->ele_num_last_core_input_scalar,
                           params->ele_num_full_mask_full_repeat_time_input_scalar, params->last_core_input_scalar);
    } else if (params->need_core_num_input_scalar > 1) {
      // use all core
      // front core
      params->ele_num_front_core_input_scalar = ele_num_fp32 / params->need_core_num_input_scalar;
      ComputeParamsOneCore(params->ele_num_front_core_input_scalar,
                           params->ele_num_full_mask_full_repeat_time_input_scalar, params->front_core_input_scalar);
      // last core
      params->ele_num_last_core_input_scalar =
          ele_num_fp32 - params->ele_num_front_core_input_scalar * (params->need_core_num_input_scalar - 1);
      ComputeParamsOneCore(params->ele_num_last_core_input_scalar,
                           params->ele_num_full_mask_full_repeat_time_input_scalar, params->last_core_input_scalar);
    }
    context->SetBlockDim(params->need_core_num_input_scalar);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDynamicAtomicAddrClean(gert::TilingParseContext* context) {
  auto compile_info = MutableCompileInfo<DynamicAtomicAddrCleanCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get vars failed."),
                  return ge::GRAPH_FAILED);
  GetCompileValue(*parsed_object_cinfo, "_workspace_size_list", compile_info->_workspace_size_list);
  GetCompileValue(vars, "ub_size", compile_info->ub_size);
  GetCompileValue(vars, "core_num", compile_info->core_num);
  GetCompileValue(vars, "workspace_num", compile_info->workspace_num);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(DynamicAtomicAddrClean)
    .Tiling(TilingForDynamicAtomicAddrClean)
    .TilingParse<DynamicAtomicAddrCleanCompileInfo>(TilingPrepareForDynamicAtomicAddrClean);
}  // namespace optiling
