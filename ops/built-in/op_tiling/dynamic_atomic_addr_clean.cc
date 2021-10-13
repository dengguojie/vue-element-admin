/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file dynamic_atomic_addr_clean.cpp
 * \brief
 */
#include <vector>
#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"

#include "op_log.h"
#include "error_log.h"
#include "graph/utils/op_desc_utils.h"
#include "vector_tiling_profiling.h"

namespace optiling {

const string LOG_INFO = "[INFO] [AtomicAddrClean] ";
const uint32_t MIN_ELE_SIZE_USING_ALL_CORE = 1024;
const uint32_t BYTE_BLOCK = 32;
const uint32_t BYTE_FP32 = 4;
const uint32_t MASK_FP32 = 64;
const uint32_t MAX_REPEAT_TIME = 255;

struct CleanTilingParams {
  // common input scalar
  int32_t select_key_input_scalar;
  int32_t need_core_num_input_scalar;
  int32_t ele_num_full_mask_full_repeat_time_input_scalar;
  int32_t burst_len_full_mask_full_repeat_time_input_scalar;

  // init input scalar
  // front core
  int32_t ele_num_front_core_input_scalar;
  int32_t init_times_full_mask_full_repeat_time_front_core_input_scalar;
  int32_t ele_num_front_part_front_core_input_scalar;
  int32_t burst_len_last_part_front_core_input_scalar;
  int32_t repeat_time_last_part_front_core_input_scalar;
  // last core
  int32_t ele_num_last_core_input_scalar;
  int32_t init_times_full_mask_full_repeat_time_last_core_input_scalar;
  int32_t ele_num_front_part_last_core_input_scalar;
  int32_t burst_len_last_part_last_core_input_scalar;
  int32_t repeat_time_last_part_last_core_input_scalar;
};

int32_t CeilDiv(const int32_t& num, const int32_t& factor) {
  int32_t res;
  res = (num % factor == 0) ? num / factor : num / factor + 1;
  return res;
}

bool GetCompileParams(const std::string& op_type, const nlohmann::json& opCompileInfoJson, uint32_t& workspace_num,
                      uint32_t& core_num, uint32_t& ub_size) {
  using namespace nlohmann;
  const auto& all_vars = opCompileInfoJson["vars"];
  if (all_vars.count("workspace_num") != 0) {
    // om compatibility, so increase the default value
    workspace_num = all_vars["workspace_num"].get<std::uint32_t>();
    OP_LOGI(op_type.c_str(), "get workspace_num sucess");
  }
  if (all_vars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get core_num failed");
    return false;
  }
  core_num = all_vars["core_num"].get<std::uint32_t>();
  if (all_vars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get ub_size failed");
    return false;
  }
  ub_size = all_vars["ub_size"].get<std::uint32_t>();
  return true;
}

void ComputeParamsOneCore(const int32_t& ele_num_one_core,
                          const int32_t& ele_num_full_mask_full_repeat_time_input_scalar,
                          int32_t& init_times_full_mask_full_repeat_time_input_scalar,
                          int32_t& ele_num_front_part_input_scalar, int32_t& burst_len_last_part_input_scalar,
                          int32_t& repeat_time_last_part_input_scalar) {
  init_times_full_mask_full_repeat_time_input_scalar =
      ele_num_one_core / ele_num_full_mask_full_repeat_time_input_scalar;
  ele_num_front_part_input_scalar =
      init_times_full_mask_full_repeat_time_input_scalar * ele_num_full_mask_full_repeat_time_input_scalar;
  uint32_t ele_num_last_part = ele_num_one_core - ele_num_front_part_input_scalar;
  burst_len_last_part_input_scalar = CeilDiv(ele_num_last_part * BYTE_FP32, BYTE_BLOCK);
  if (ele_num_last_part % MASK_FP32 == 0) {
    repeat_time_last_part_input_scalar = ele_num_last_part / MASK_FP32;
  } else {
    repeat_time_last_part_input_scalar = ele_num_last_part / MASK_FP32 + 1;
  }
}

void InitTilingParams(CleanTilingParams& params) {
  params.select_key_input_scalar = 0;
  params.need_core_num_input_scalar = 0;
  params.ele_num_full_mask_full_repeat_time_input_scalar = 0;
  params.burst_len_full_mask_full_repeat_time_input_scalar = 0;

  // init input scalar
  // front core
  params.ele_num_front_core_input_scalar = 0;
  params.init_times_full_mask_full_repeat_time_front_core_input_scalar = 0;
  params.ele_num_front_part_front_core_input_scalar = 0;
  params.burst_len_last_part_front_core_input_scalar = 0;
  params.repeat_time_last_part_front_core_input_scalar = 0;
  // last core
  params.ele_num_last_core_input_scalar = 0;
  params.init_times_full_mask_full_repeat_time_last_core_input_scalar = 0;
  params.ele_num_front_part_last_core_input_scalar = 0;
  params.burst_len_last_part_last_core_input_scalar = 0;
  params.repeat_time_last_part_last_core_input_scalar = 0;
}

void WriteTilingParams(const CleanTilingParams& params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(params.select_key_input_scalar);
  run_info.AddTilingData(params.need_core_num_input_scalar);
  run_info.AddTilingData(params.ele_num_full_mask_full_repeat_time_input_scalar);
  run_info.AddTilingData(params.burst_len_full_mask_full_repeat_time_input_scalar);
  run_info.AddTilingData(params.ele_num_front_core_input_scalar);
  run_info.AddTilingData(params.init_times_full_mask_full_repeat_time_front_core_input_scalar);
  run_info.AddTilingData(params.ele_num_front_part_front_core_input_scalar);
  run_info.AddTilingData(params.burst_len_last_part_front_core_input_scalar);
  run_info.AddTilingData(params.repeat_time_last_part_front_core_input_scalar);
  run_info.AddTilingData(params.ele_num_last_core_input_scalar);
  run_info.AddTilingData(params.init_times_full_mask_full_repeat_time_last_core_input_scalar);
  run_info.AddTilingData(params.ele_num_front_part_last_core_input_scalar);
  run_info.AddTilingData(params.burst_len_last_part_last_core_input_scalar);
  run_info.AddTilingData(params.repeat_time_last_part_last_core_input_scalar);
}

void PrintTilingParams(const std::string& op_type, const CleanTilingParams& params) {
  OP_LOGD(op_type.c_str(), "op [%s] : params.select_key_input_scalar=%d", op_type.c_str(),
          params.select_key_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.need_core_num_input_scalar=%d", op_type.c_str(),
          params.need_core_num_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.ele_num_full_mask_full_repeat_time_input_scalar=%d", op_type.c_str(),
          params.ele_num_full_mask_full_repeat_time_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.burst_len_full_mask_full_repeat_time_input_scalar=%d", op_type.c_str(),
          params.burst_len_full_mask_full_repeat_time_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.ele_num_front_core_input_scalar=%d", op_type.c_str(),
          params.ele_num_front_core_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.init_times_full_mask_full_repeat_time_front_core_input_scalar=%d",
          op_type.c_str(), params.init_times_full_mask_full_repeat_time_front_core_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.ele_num_front_part_front_core_input_scalar=%d", op_type.c_str(),
          params.ele_num_front_part_front_core_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.burst_len_last_part_front_core_input_scalar=%d", op_type.c_str(),
          params.burst_len_last_part_front_core_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.repeat_time_last_part_front_core_input_scalar=%d", op_type.c_str(),
          params.repeat_time_last_part_front_core_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.ele_num_last_core_input_scalar=%d", op_type.c_str(),
          params.ele_num_last_core_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.init_times_full_mask_full_repeat_time_last_core_input_scalar=%d",
          op_type.c_str(), params.init_times_full_mask_full_repeat_time_last_core_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.ele_num_front_part_last_core_input_scalar=%d", op_type.c_str(),
          params.ele_num_front_part_last_core_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.burst_len_last_part_last_core_input_scalar=%d", op_type.c_str(),
          params.burst_len_last_part_last_core_input_scalar);
  OP_LOGD(op_type.c_str(), "op [%s] : params.repeat_time_last_part_last_core_input_scalar=%d", op_type.c_str(),
          params.repeat_time_last_part_last_core_input_scalar);
}

bool CheckSize(const std::string& op_type, const uint32_t& size) {
  if (size < 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op: workspace size must be greater than 0!");
    return false;
  }
  if (size % 32 != 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op : workspace size must be able to be divided by 32!");
    return false;
  }
  return true;
}

// tiling function
bool DynamicAtomicAddrCleanTiling(const std::string& op_type, const ge::Operator& op_paras,
                                  const nlohmann::json& opCompileInfoJson, utils::OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "op[%s] op tiling begin.", op_type.c_str());

  PROFILING_TILING_INIT(op_type.c_str());

  std::vector<int64_t> workspace_size_list;
  uint32_t workspace_num = 1;
  uint32_t core_num = 1;
  uint32_t ub_size = 256 * 1024;
  // get compile_info params
  bool flag = GetCompileParams(op_type, opCompileInfoJson, workspace_num, core_num, ub_size);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetCompileParams failed");
    return false;
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  OP_LOGI(op_type.c_str(), "op[%s] GetCompileParams success.", op_type.c_str());
  if (opCompileInfoJson.find("_workspace_size_list") == opCompileInfoJson.end()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "key _workspace_size_list not exists.");
    return false;
  } else {
    const auto& workspace_size_list_1 = opCompileInfoJson["_workspace_size_list"];
    for (auto iter = workspace_size_list_1.begin(); iter != workspace_size_list_1.end(); iter++) {
      workspace_size_list.push_back((int64_t)*iter);
    }
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  auto size_count = workspace_size_list.size();
  for (size_t i = 0; i < size_count; i++) {
    int64_t addr_tensor_size = workspace_size_list[i];
    bool flag = CheckSize(op_type, addr_tensor_size);
    if (!flag) {
      return false;
    }
    OP_LOGD(op_type.c_str(), "op: addr_tensor_size=%d", addr_tensor_size);
    uint32_t ele_num_fp32 = addr_tensor_size / BYTE_FP32;
    CleanTilingParams params;
    // init tiling params
    InitTilingParams(params);
    params.select_key_input_scalar = 1;
    // is using all core
    if (addr_tensor_size >= MIN_ELE_SIZE_USING_ALL_CORE) {
      params.need_core_num_input_scalar = core_num;
    } else {
      params.need_core_num_input_scalar = 1;
    }
    // compute tiling params
    params.ele_num_full_mask_full_repeat_time_input_scalar = MASK_FP32 * MAX_REPEAT_TIME;
    params.burst_len_full_mask_full_repeat_time_input_scalar =
        params.ele_num_full_mask_full_repeat_time_input_scalar * BYTE_FP32 / BYTE_BLOCK;
    if (params.need_core_num_input_scalar == 1) {
      // use one core
      params.ele_num_front_core_input_scalar = ele_num_fp32;
      ComputeParamsOneCore(
          params.ele_num_front_core_input_scalar, params.ele_num_full_mask_full_repeat_time_input_scalar,
          params.init_times_full_mask_full_repeat_time_front_core_input_scalar,
          params.ele_num_front_part_front_core_input_scalar, params.burst_len_last_part_front_core_input_scalar,
          params.repeat_time_last_part_front_core_input_scalar);

      params.ele_num_last_core_input_scalar = params.ele_num_front_core_input_scalar;
      ComputeParamsOneCore(
          params.ele_num_last_core_input_scalar, params.ele_num_full_mask_full_repeat_time_input_scalar,
          params.init_times_full_mask_full_repeat_time_last_core_input_scalar,
          params.ele_num_front_part_last_core_input_scalar, params.burst_len_last_part_last_core_input_scalar,
          params.repeat_time_last_part_last_core_input_scalar);
    } else if (params.need_core_num_input_scalar > 1) {
      // use all core
      // front core
      params.ele_num_front_core_input_scalar = ele_num_fp32 / params.need_core_num_input_scalar;
      ComputeParamsOneCore(
          params.ele_num_front_core_input_scalar, params.ele_num_full_mask_full_repeat_time_input_scalar,
          params.init_times_full_mask_full_repeat_time_front_core_input_scalar,
          params.ele_num_front_part_front_core_input_scalar, params.burst_len_last_part_front_core_input_scalar,
          params.repeat_time_last_part_front_core_input_scalar);
      // last core
      params.ele_num_last_core_input_scalar =
          ele_num_fp32 - params.ele_num_front_core_input_scalar * (params.need_core_num_input_scalar - 1);
      ComputeParamsOneCore(
          params.ele_num_last_core_input_scalar, params.ele_num_full_mask_full_repeat_time_input_scalar,
          params.init_times_full_mask_full_repeat_time_last_core_input_scalar,
          params.ele_num_front_part_last_core_input_scalar, params.burst_len_last_part_last_core_input_scalar,
          params.repeat_time_last_part_last_core_input_scalar);
    }
    // write tiling params to run_info
    WriteTilingParams(params, run_info);
    PROFILING_TILING_AFTER_CALCU_TILING_REG();
    // print tiling params
    PrintTilingParams(op_type, params);
    // block_dim, core num used in tik op
    run_info.SetBlockDim(params.need_core_num_input_scalar);
    PROFILING_TILING_END();
  }
  OP_LOGI(op_type.c_str(), "op[%s] op tiling success", op_type.c_str());
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED_V2(DynamicAtomicAddrClean, DynamicAtomicAddrCleanTiling);
}  // namespace optiling
