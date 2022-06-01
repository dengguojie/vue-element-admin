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
#include "runtime/atomic_clean_tiling_context.h"

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

static void InitTilingParams(DynamicAtomicAddrCleanTilingData *params, size_t total_clean_num) {
  size_t len = sizeof(DynamicAtomicAddrCleanTilingData) * total_clean_num;
  memset_s(params, len, 0, len);
}

ge::graphStatus WriteAtomicTilingData(gert::TilingContext *context, DynamicAtomicAddrCleanTilingData *params,
                                      int64_t tensor_size, uint32_t core_num) {
  OP_TILING_CHECK(
    ((tensor_size < 0) || (tensor_size % 32 != 0)),
    VECTOR_INNER_ERR_REPORT_TILIING(
        context->GetNodeName(), "tensor_size %ld error! must be a natural multiple of 32", tensor_size),
    return ge::GRAPH_FAILED);
  uint32_t ele_num_fp32 = tensor_size / BYTE_FP32;
  params->select_key_input_scalar = 1;
  // is using all core
  if (tensor_size >= MIN_ELE_SIZE_USING_ALL_CORE) {
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
    ComputeParamsOneCore(
        params->ele_num_front_core_input_scalar, params->ele_num_full_mask_full_repeat_time_input_scalar,
        params->front_core_input_scalar);

    params->ele_num_last_core_input_scalar = params->ele_num_front_core_input_scalar;
    ComputeParamsOneCore(
        params->ele_num_last_core_input_scalar, params->ele_num_full_mask_full_repeat_time_input_scalar,
        params->last_core_input_scalar);
  } else if (params->need_core_num_input_scalar > 1) {
    // use all core
    // front core
    params->ele_num_front_core_input_scalar = ele_num_fp32 / params->need_core_num_input_scalar;
    ComputeParamsOneCore(
        params->ele_num_front_core_input_scalar, params->ele_num_full_mask_full_repeat_time_input_scalar,
        params->front_core_input_scalar);
    // last core
    params->ele_num_last_core_input_scalar =
        ele_num_fp32 - params->ele_num_front_core_input_scalar * (params->need_core_num_input_scalar - 1);
    ComputeParamsOneCore(
        params->ele_num_last_core_input_scalar, params->ele_num_full_mask_full_repeat_time_input_scalar,
        params->last_core_input_scalar);
  }
  context->SetBlockDim(params->need_core_num_input_scalar);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForDynamicAtomicAddrClean(gert::TilingContext* context) {
  auto compile_info = reinterpret_cast<const DynamicAtomicAddrCleanCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  auto compute_node_info = context->GetComputeNodeInfo();
  OPS_CHECK_NULL_WITH_CONTEXT(context, compute_node_info);

  uint32_t core_num = compile_info->core_num;
  const std::vector<int64_t> &workspace_idx = compile_info->_workspace_index_list;

  size_t clean_tensor_num = compute_node_info->GetInputsNum() - 1;
  size_t total_clean_num = clean_tensor_num + workspace_idx.size();

  gert::TilingData *tiling_data = context->GetRawTilingData();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tiling_data);
  tiling_data->SetDataSize(sizeof(DynamicAtomicAddrCleanTilingData) * total_clean_num);
  auto params = reinterpret_cast<DynamicAtomicAddrCleanTilingData *>(tiling_data->GetData());
  // init tiling params
  InitTilingParams(params, total_clean_num);
  auto atomic_clean_context = reinterpret_cast<gert::AtomicCleanTilingContext *>(context);

  for (size_t idx = 0U; idx < clean_tensor_num; ++idx, ++params) {
    auto tensor_size = atomic_clean_context->GetCleanOutputSize(idx);
    if (WriteAtomicTilingData(context, params, tensor_size, core_num) != ge::GRAPH_SUCCESS) {
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "op: write atomic tiling data failed!");
      return ge::GRAPH_FAILED;
    }
  }

  if (!workspace_idx.empty()) {
    auto ws_sizes = atomic_clean_context->GetCleanWorkspaceSizes();
    OPS_CHECK_NULL_WITH_CONTEXT(context, ws_sizes);
    auto ws_size_data = reinterpret_cast<const uint64_t *>(ws_sizes->GetData());
    for (size_t i = 0U; i < workspace_idx.size(); ++i, ++params) {
      auto tensor_size = ws_size_data[workspace_idx[i]];
      if (WriteAtomicTilingData(context, params, tensor_size, core_num) != ge::GRAPH_SUCCESS) {
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "op: write atomic tiling data failed!");
        return ge::GRAPH_FAILED;
      }
    }
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
  GetCompileValue(*parsed_object_cinfo, "_workspace_index_list", compile_info->_workspace_index_list);
  GetCompileValue(vars, "ub_size", compile_info->ub_size);
  GetCompileValue(vars, "core_num", compile_info->core_num);
  GetCompileValue(vars, "workspace_num", compile_info->workspace_num);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(DynamicAtomicAddrClean)
    .Tiling(TilingForDynamicAtomicAddrClean)
    .TilingParse<DynamicAtomicAddrCleanCompileInfo>(TilingPrepareForDynamicAtomicAddrClean);
}  // namespace optiling
