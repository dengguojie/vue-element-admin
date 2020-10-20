/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file scatter_sub.cpp
 * \brief tiling function of scattersub
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "register/op_tiling.h"
#include "graph/debug/ge_log.h"

#include "op_log.h"

const uint32_t MAX_UB_UPDATES = 474 * 128;
const uint32_t MAX_UB_CORE_INDICES = 474;
// for the cloud platform fp32 data
const int32_t TILING_MODE_1 = 1;
// deal one core data less than MAX_UB_CORE_INDICES
const int32_t SELECT_LESS_THAN_PARAMS = 0;
// deal one core data more than MAX_UB_CORE_INDICES
const int32_t SELECT_MORE_THAN_PARAMS = 1;

namespace optiling {
/*
 * @brief: tiling function of scattersub
 * @param [in] opType: opType of the scattersub
 * @param [in] opParas: inputs/outputs/atts of the scattersub
 * @param [in] opCompileInfo: compile time generated info of the scattersub
 * @param [out] runInfo: result data
 * @return bool: success or not
 */

const uint32_t BYTE_BLOCK = 32;
typedef enum { FP16_BYTE = 2, FP32_BYTE = 4, INT32_BYTE = 4 } EleByte;

struct ScatterSub {
  uint32_t select_mode;
  uint32_t select_params;
  uint32_t final_one_core_data_num;
  uint32_t last_core_data_num;
  uint32_t indices_ub_number;
  uint32_t core_num;
  uint32_t updates_data_num;
  uint32_t updates_burst_fact_len;
  uint32_t indices_burst_len;
  uint32_t updates_burst_len;
  uint32_t block_number;
  uint32_t tail_indices_burst_len;
  uint32_t tail_indices_num_burst_len;
  uint32_t tail_updates_burst_len;
  uint32_t tail_indices_more_than_burst_len;
  uint32_t tail_updates_can_div;
  uint32_t select_align_params;
  uint32_t max_align_updates_data_num;
};

bool GetScatterSubParams(const std::string& opType, const nlohmann::json& opCompileInfo, uint32_t& coreNum,
                         uint32_t& ubSize) {
  using namespace nlohmann;
  auto allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    OP_LOGE(opType.c_str(), "op [ScatterSubTiling] : GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::uint32_t>();
  if (allVars.count("ub_size") == 0) {
    OP_LOGE(opType.c_str(), "op [ScatterSubTiling] : GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::uint32_t>();
  return true;
}

void TilingScatterSubIndicesCore(uint32_t& coreNum, uint32_t& inputSize, uint32_t& one_core_data,
                                 uint32_t& last_core_data_num) {
  uint32_t one_core_data_num = 0;
  uint32_t add_sum = 0;
  uint32_t sub_sum = 0;
  if (inputSize % coreNum == 0) {
    one_core_data_num = inputSize / coreNum;
  } else {
    one_core_data_num = (inputSize / coreNum) + 1;
  }
  const uint32_t& pre_core_data_num = one_core_data_num * (coreNum - 1);
  if (pre_core_data_num <= inputSize) {
    for (uint32_t i = 0; i <= one_core_data_num; i++) {
      const uint32_t& last_data_num = one_core_data_num + i;
      const uint32_t& last_data_num_one = last_data_num * (coreNum - 1);
      if (last_data_num_one < inputSize) {
        add_sum = i;
        break;
      }
    }
    one_core_data = one_core_data_num + add_sum;
    last_core_data_num = inputSize - (coreNum - 1) * one_core_data;
  } else {
    for (uint32_t i = 0; i <= one_core_data_num; i++) {
      const uint32_t& sub_data_num = one_core_data_num - i;
      const uint32_t& sub_data_num_one = sub_data_num * (coreNum - 1);
      if (sub_data_num_one < inputSize) {
        sub_sum = i;
        break;
      }
    }
    one_core_data = one_core_data_num - sub_sum;
    last_core_data_num = inputSize - (coreNum - 1) * one_core_data;
  }
}

void InitRunningParams(ScatterSub& params) {
  params.select_mode = 0;
  params.select_params = 0;
  params.final_one_core_data_num = 0;
  params.last_core_data_num = 0;
  params.indices_ub_number = 0;
  params.core_num = 0;
  params.updates_data_num = 0;
  params.updates_burst_fact_len = 0;
  params.indices_burst_len = 0;
  params.updates_burst_len = 0;
  params.block_number = 0;
  params.tail_indices_burst_len = 0;
  params.tail_indices_num_burst_len = 0;
  params.tail_updates_burst_len = 0;
  params.tail_indices_more_than_burst_len = 0;
  params.tail_updates_can_div = 0;
  params.select_align_params = 0;
  params.max_align_updates_data_num = 0;
}

void SetRuningParams(const ScatterSub& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.select_mode);
  ByteBufferPut(runInfo.tiling_data, params.select_params);
  ByteBufferPut(runInfo.tiling_data, params.final_one_core_data_num);
  ByteBufferPut(runInfo.tiling_data, params.last_core_data_num);
  ByteBufferPut(runInfo.tiling_data, params.indices_ub_number);
  ByteBufferPut(runInfo.tiling_data, params.core_num);
  ByteBufferPut(runInfo.tiling_data, params.updates_data_num);
  ByteBufferPut(runInfo.tiling_data, params.updates_burst_fact_len);
  ByteBufferPut(runInfo.tiling_data, params.indices_burst_len);
  ByteBufferPut(runInfo.tiling_data, params.updates_burst_len);
  ByteBufferPut(runInfo.tiling_data, params.block_number);
  ByteBufferPut(runInfo.tiling_data, params.tail_indices_burst_len);
  ByteBufferPut(runInfo.tiling_data, params.tail_indices_num_burst_len);
  ByteBufferPut(runInfo.tiling_data, params.tail_updates_burst_len);
  ByteBufferPut(runInfo.tiling_data, params.tail_indices_more_than_burst_len);
  ByteBufferPut(runInfo.tiling_data, params.tail_updates_can_div);
  ByteBufferPut(runInfo.tiling_data, params.select_align_params);
  ByteBufferPut(runInfo.tiling_data, params.max_align_updates_data_num);
}

void PrintParams(const ScatterSub& params) {
  GELOGD("params.select_mode is %d", params.select_mode);
  GELOGD("params.select_params %d", params.select_params);
  GELOGD("params.final_one_core_data_num is %d", params.final_one_core_data_num);
  GELOGD("params.last_core_data_num is %d", params.last_core_data_num);
  GELOGD("params.indices_ub_number is %d", params.indices_ub_number);
  GELOGD("params.core_num is %d", params.core_num);
  GELOGD("params.updates_data_num is %d", params.updates_data_num);
  GELOGD("params.updates_burst_fact_len is %d", params.updates_burst_fact_len);
  GELOGD("params.indices_burst_len is %d", params.indices_burst_len);
  GELOGD("params.updates_burst_len is %d", params.updates_burst_len);
  GELOGD("params.block_number is %d", params.block_number);
  GELOGD("params.tail_indices_burst_len is %d", params.tail_indices_burst_len);
  GELOGD("params.tail_indices_num_burst_len is %d", params.tail_indices_num_burst_len);
  GELOGD("params.tail_updates_burst_len is %d", params.tail_updates_burst_len);
  GELOGD("params.tail_indices_more_than_burst_len is %d", params.tail_indices_more_than_burst_len);
  GELOGD("params.tail_updates_can_div is %d", params.tail_updates_can_div);
  GELOGD("params.select_align_params is %d", params.select_align_params);
  GELOGD("params.max_align_updates_data_num is %d", params.max_align_updates_data_num);
}

bool ScatterSubTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                      OpRunInfo& runInfo) {
  using namespace nlohmann;
  GELOGI("op[%s] ScatterSubTiling running.", opType.c_str());
  const std::vector<int64_t>& inputShape1 = opParas.inputs[1].tensor[0].shape;
  const std::vector<int64_t>& inputShape2 = opParas.inputs[2].tensor[0].shape;
  ScatterSub params;
  InitRunningParams(params);

  params.indices_ub_number = std::accumulate(inputShape1.begin(), inputShape1.end(), 1, std::multiplies<int>());
  const uint32_t& updatesize = std::accumulate(inputShape2.begin(), inputShape2.end(), 1, std::multiplies<int>());

  const std::string& update_dtype = opParas.inputs[2].tensor[0].dtype;
  EleByte input_update_byte;
  if (update_dtype == "float32" || update_dtype == "int32") {
    input_update_byte = FP32_BYTE;
  } else {
    input_update_byte = FP16_BYTE;
  }
  params.block_number = BYTE_BLOCK / input_update_byte;
  uint32_t ubSize;
  uint32_t coreNum;
  bool flag = GetScatterSubParams(opType, opCompileInfo, coreNum, ubSize);
  if (!flag) {
    return false;
  }

  params.updates_data_num = updatesize / params.indices_ub_number;
  if (params.updates_data_num % params.block_number == 0) {
    params.updates_burst_fact_len = params.updates_data_num / params.block_number;
    params.select_align_params = 0;
    params.max_align_updates_data_num = 0;
  } else {
    params.select_align_params = 1;
    params.updates_burst_fact_len = ceil(float(params.updates_data_num) / params.block_number);
    if (params.updates_data_num > params.block_number) {
      params.max_align_updates_data_num = (params.updates_data_num / params.block_number) * params.block_number;
      if (params.updates_data_num > MAX_UB_UPDATES) {
        const uint32_t& tail_loop_updates_data_num = params.updates_data_num % MAX_UB_UPDATES;
        params.tail_updates_burst_len = ceil(float(tail_loop_updates_data_num) / params.block_number);
        params.tail_updates_can_div = (tail_loop_updates_data_num / params.block_number) * params.block_number;
      }
    }
  }

  if (params.indices_ub_number < BYTE_BLOCK) {
    params.select_mode = TILING_MODE_1;
    params.select_params = SELECT_LESS_THAN_PARAMS;
    params.core_num = 1;
    params.indices_burst_len = ceil(float(params.indices_ub_number) / params.block_number);
  }
  if (params.indices_ub_number >= BYTE_BLOCK) {
    uint32_t final_one_core_data_num_one = 0;
    uint32_t last_core_data_num_one = 0;
    TilingScatterSubIndicesCore(coreNum, params.indices_ub_number, final_one_core_data_num_one, last_core_data_num_one);
    params.final_one_core_data_num = final_one_core_data_num_one;
    params.last_core_data_num = last_core_data_num_one;
    params.core_num = coreNum;
    params.indices_burst_len = ceil(float(params.final_one_core_data_num) / params.block_number);
    params.tail_indices_burst_len = ceil(float(params.last_core_data_num) / params.block_number);
    if (params.final_one_core_data_num <= MAX_UB_CORE_INDICES) {
      params.select_mode = TILING_MODE_1;
      params.select_params = SELECT_LESS_THAN_PARAMS;
      params.updates_burst_len =
          ceil(float(params.final_one_core_data_num * params.updates_data_num) / params.block_number);
      params.tail_updates_burst_len =
          ceil(float(params.last_core_data_num * params.updates_data_num) / params.block_number);
      if (params.last_core_data_num > MAX_UB_CORE_INDICES) {
        const uint32_t& tail_inidces_number = params.last_core_data_num % MAX_UB_CORE_INDICES;
        params.tail_indices_num_burst_len = ceil(float(tail_inidces_number) / params.block_number);
      }
    } else {
      params.select_mode = TILING_MODE_1;
      params.select_params = SELECT_MORE_THAN_PARAMS;
      const uint32_t& tail_indices_number = params.final_one_core_data_num % MAX_UB_CORE_INDICES;
      params.tail_indices_num_burst_len = ceil(float(tail_indices_number) / params.block_number);
      if (params.last_core_data_num > MAX_UB_CORE_INDICES) {
        const uint32_t& tail_last_indices_number = params.last_core_data_num % MAX_UB_CORE_INDICES;
        params.tail_indices_more_than_burst_len = ceil(float(tail_last_indices_number) / params.block_number);
      }
    }
  }

  SetRuningParams(params, runInfo);
  PrintParams(params);

  runInfo.block_dim = params.core_num;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  GELOGI("op[%s] tiling run success.", opType.c_str());
  return true;
}

REGISTER_OP_TILING_FUNC(ScatterSub, ScatterSubTiling);
}  // namespace optiling
