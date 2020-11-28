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
 * \file scatter_nd.cpp
 * \brief tiling function of scatternd
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "op_tiling.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"

const uint32_t MAX_UB_CORE_INDICES = 474;
const uint32_t MAX_UB_UPDATES = 474 * 128;
const int32_t LAST_DIM_MAX = 8;

// for the cloud platform fp32 data
const int32_t TILING_MODE_1 = 1;
// deal one core data less than MAX_UB_CORE_INDICES
const int32_t SELECT_LESS_THAN_PARAMS = 0;
// deal one core data more than MAX_UB_CORE_INDICES
const int32_t SELECT_MORE_THAN_PARAMS = 1;
const uint32_t MAX_INT32_FP32_SHAPE_UB = 220 * 128;

namespace optiling {
/*
 * @brief: tiling function of scatternd
 * @param [in] opType: opType of the scatternd
 * @param [in] opParas: inputs/outputs/atts of the scatternd
 * @param [in] opCompileInfo: compile time generated info of the scatternd
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
const uint32_t BYTE_BLOCK = 32;
typedef enum { FP16_BYTE = 2, FP32_BYTE = 4, INT32_BYTE = 4 } EleByte;

struct ScatterNd {
  uint32_t select_mode;
  uint32_t select_params;
  uint32_t indices_num;
  uint32_t core_num;
  uint32_t one_core_data;
  uint32_t last_core_data_num;
  uint32_t block_number;
  uint32_t indices_num_one_burst_len;
  uint32_t updates_num_one_burst_len;
  uint32_t updates_data_num;
  uint32_t updates_burst_fact_len;
  uint32_t tail_indices_burst_len;
  uint32_t tail_updates_burst_len;
  uint32_t tail_updates_can_div;
  uint32_t tail_indices_num_burst_len;
  uint32_t tail_indices_more_than_burst_len;
  uint32_t select_align_params;
  uint32_t max_align_updates_data_num;
};

bool CheckScatterNdTensorShape(const std::string& opType, std::vector<int64_t> indicesShape,
                               std::vector<int64_t> updatesShape, std::vector<int64_t> outputShape,
                               int32_t indicesLastDim) {
  const int32_t& indicesDims = indicesShape.size();
  const int32_t& updatesDims = updatesShape.size();
  const int32_t& outputDims = outputShape.size();

  if (indicesLastDim == 0 && indicesDims == 1) {
    ge::OpsOneInputShapeErrReport("ScatterNd", "indices",
                                  "the length of indices is one and the last dim of indices's shape is zero");
    OP_LOGE(opType.c_str(), "op ScatterNdTiling : CheckTensorShape, indices's shape is invalid");
    return false;
  }
  for (int32_t i = 0; i < indicesDims - 1; i++) {
    if (indicesShape[i] <= 0) {
      ge::OpsOneInputShapeErrReport("ScatterNd", "indices", "the indices's shape should be less than zero");
      OP_LOGE(opType.c_str(), "op ScatterNdTiling : the indices's shape must be more than zero");
      return false;
    }
  }
  for (int32_t i = 0; i < indicesDims - 1; i++) {
    if (indicesShape[i] != updatesShape[i]) {
      ge::OpsOneInputShapeErrReport("ScatterNd", "indices",
                                    "indices's shape and updates'shape are not equal in some dimensions");
      OP_LOGE(opType.c_str(),
              "op ScatterNdTiling : indices's shape and update's shape must be same in some dimensions");
      return false;
    }
  }
  if (indicesDims - 1 + outputDims - indicesLastDim != updatesDims) {
    ge::OpsOneInputShapeErrReport("ScatterNd", "updates",
                                  "output's shape and updates'shape are not equal in some dimensions");
    OP_LOGE(opType.c_str(), "op ScatterNdTiling : update's shape and output's shape must be same in some dimension");
    return false;
  }

  for (int32_t i = 0; i < updatesDims - indicesDims + 1; i++) {
    if (updatesShape[indicesDims - 1 + i] != outputShape[indicesLastDim + i]) {
      ge::OpsOneInputShapeErrReport("ScatterNd", "updates",
                                    "output's shape and updates'shape are not equal in some dimensions");
      OP_LOGE(opType.c_str(), "op ScatterNdTiling : update's shape and output's shape must be same in some dimension");

      return false;
    }
  }
  return true;
}

bool GetSocParamsTwo(const std::string& opType, const nlohmann::json& opCompileInfo, uint32_t& coreNum,
                     uint32_t& ubSize) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "core_num");
    OP_LOGE(opType.c_str(), "op ScatterNdTiling : GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::uint32_t>();
  if (allVars.count("ub_size") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "ub_size");
    OP_LOGE(opType.c_str(), "op ScatterNdTiling : GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::uint32_t>();
  return true;
}

void InitRunningParams(ScatterNd& params) {
  params.select_mode = 0;
  params.select_params = 0;
  params.indices_num = 0;
  params.core_num = 0;
  params.one_core_data = 0;
  params.last_core_data_num = 0;
  params.block_number = 0;
  params.indices_num_one_burst_len = 0;
  params.updates_num_one_burst_len = 0;
  params.updates_data_num = 0;
  params.updates_burst_fact_len = 0;
  params.tail_indices_burst_len = 0;
  params.tail_updates_burst_len = 0;
  params.tail_updates_can_div = 0;
  params.tail_indices_num_burst_len = 0;
  params.tail_indices_more_than_burst_len = 0;
  params.select_align_params = 0;
  params.max_align_updates_data_num = 0;
}

void SetRunningParams(const ScatterNd& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.select_mode);
  ByteBufferPut(runInfo.tiling_data, runParams.select_params);
  ByteBufferPut(runInfo.tiling_data, runParams.indices_num);
  ByteBufferPut(runInfo.tiling_data, runParams.core_num);
  ByteBufferPut(runInfo.tiling_data, runParams.one_core_data);
  ByteBufferPut(runInfo.tiling_data, runParams.last_core_data_num);
  ByteBufferPut(runInfo.tiling_data, runParams.block_number);
  ByteBufferPut(runInfo.tiling_data, runParams.indices_num_one_burst_len);
  ByteBufferPut(runInfo.tiling_data, runParams.updates_num_one_burst_len);
  ByteBufferPut(runInfo.tiling_data, runParams.updates_data_num);
  ByteBufferPut(runInfo.tiling_data, runParams.updates_burst_fact_len);
  ByteBufferPut(runInfo.tiling_data, runParams.tail_indices_burst_len);
  ByteBufferPut(runInfo.tiling_data, runParams.tail_updates_burst_len);
  ByteBufferPut(runInfo.tiling_data, runParams.tail_updates_can_div);
  ByteBufferPut(runInfo.tiling_data, runParams.tail_indices_num_burst_len);
  ByteBufferPut(runInfo.tiling_data, runParams.tail_indices_more_than_burst_len);
  ByteBufferPut(runInfo.tiling_data, runParams.select_align_params);
  ByteBufferPut(runInfo.tiling_data, runParams.max_align_updates_data_num);
}

void ScatterCoreSplit(uint32_t& coreNum, uint32_t& inputSize, uint32_t& one_core_data, uint32_t& last_core_data_num) {
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
    for (size_t i = 0; i <= one_core_data_num; i++) {
      const uint32_t& last_data_num_one = (one_core_data_num + i) * (coreNum - 1);
      if (last_data_num_one < inputSize) {
        add_sum = i;
        break;
      }
    }
    one_core_data = one_core_data_num + add_sum;
    last_core_data_num = inputSize - (coreNum - 1) * one_core_data;
  } else {
    for (size_t i = 0; i <= one_core_data_num; i++) {
      const uint32_t& sub_data_num_one = (one_core_data_num - i) * (coreNum - 1);
      if (sub_data_num_one < inputSize) {
        sub_sum = i;
        break;
      }
    }
    one_core_data = one_core_data_num - sub_sum;
    last_core_data_num = inputSize - (coreNum - 1) * one_core_data;
  }
}

bool ScatterNdTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                     OpRunInfo& runInfo) {
  using namespace nlohmann;
  OP_LOGI(opType.c_str(), "op ScatterNdTiling running.");
  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty() || opParas.inputs[1].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices or updates", "the input may be empty");
    OP_LOGE(opType.c_str(), "op ScatterNdTiling: input shape error");
    return false;
  }
  if (opParas.outputs.empty() || opParas.outputs[0].tensor.empty()) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "out", "the output may be empty");
    OP_LOGE(opType.c_str(), "op ScatterNdTiling: output shape error");
    return false;
  }
  const std::vector<int64_t>& inputShape0 = opParas.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& inputShape1 = opParas.inputs[1].tensor[0].shape;
  const std::vector<int64_t>& outputShape = opParas.outputs[0].tensor[0].shape;

  const int32_t& outputDims = outputShape.size();
  const int32_t& indicesDims = inputShape0.size();
  // check indicesDim
  if (indicesDims <= 1) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices", "the ndim of indices is less than 1 or equal as 1");
    OP_LOGE(opType.c_str(), "op ScatterNdTiling: indices dim is invalid.");
    return false;
  }
  const int32_t& indicesLastDim = inputShape0[indicesDims - 1];
  if (indicesLastDim > outputDims || indicesLastDim > LAST_DIM_MAX || indicesLastDim < 0) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices",
                                  "the last dim of indices is more than the length of output'shape or"
                                  "the last dim of indices is greater than 8 or less than 0");
    OP_LOGE(opType.c_str(), "op ScatterNdTiling: the last dim of indices shape is invalid.");
    return false;
  }
  bool ret = CheckScatterNdTensorShape(opType, inputShape0, inputShape1, outputShape, indicesLastDim);
  if (!ret) {
    OP_LOGE(opType.c_str(), "op ScatterNdTiling : checktensorshape is failed");
    return false;
  }

  // get complie info
  uint32_t ubSize = 0;
  uint32_t coreNum = 0;
  bool flag = GetSocParamsTwo(opType, opCompileInfo, coreNum, ubSize);
  if (!flag) {
    OP_LOGE(opType.c_str(), "op ScatterNdTiling: GetCompileParams error.");
    return false;
  }

  const uint32_t& input_indice_size =
      std::accumulate(inputShape0.begin(), inputShape0.end(), 1, std::multiplies<int>());
  const uint32_t& input_update_size =
      std::accumulate(inputShape1.begin(), inputShape1.end(), 1, std::multiplies<int>());
  if (opParas.const_inputs.find("shape") == opParas.const_inputs.end()) {
    OP_LOGE(opType.c_str(), "op ScatterNdTiling: get const input failed.");
    return false;
  }

  EleByte input_update_byte;
  const std::string& input_update_dtype = opParas.inputs[1].tensor[0].dtype;
  if (input_update_dtype == "float32" || input_update_dtype == "int32") {
    input_update_byte = FP32_BYTE;
  } else {
    input_update_byte = FP16_BYTE;
  }

  ScatterNd params;
  InitRunningParams(params);
  params.block_number = BYTE_BLOCK / input_update_byte;
  params.indices_num = input_indice_size / indicesLastDim;
  params.updates_data_num = input_update_size / params.indices_num;
  if (params.updates_data_num % params.block_number == 0) {
    params.select_align_params = 0;
    params.updates_burst_fact_len = params.updates_data_num / params.block_number;
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
  // enter into fp32 branch
  if (input_update_dtype == "float32") {
    if (params.indices_num < BYTE_BLOCK) {
      params.select_mode = TILING_MODE_1;
      params.select_params = SELECT_LESS_THAN_PARAMS;
      params.core_num = 1;
      params.indices_num_one_burst_len = ceil(float(params.indices_num) / params.block_number);
    }

    if (params.indices_num >= BYTE_BLOCK) {
      uint32_t final_one_core_data_num = 0;
      uint32_t last_one_core_data_num = 0;
      ScatterCoreSplit(coreNum, params.indices_num, final_one_core_data_num, last_one_core_data_num);
      params.one_core_data = final_one_core_data_num;
      params.last_core_data_num = last_one_core_data_num;
      params.core_num = coreNum;
      params.indices_num_one_burst_len = ceil(float(params.one_core_data) / params.block_number);
      params.tail_indices_burst_len = ceil(float(params.last_core_data_num) / params.block_number);
      // deal data less than 474
      if (params.one_core_data <= MAX_UB_CORE_INDICES) {
        params.select_mode = TILING_MODE_1;
        params.select_params = SELECT_LESS_THAN_PARAMS;
        params.updates_num_one_burst_len =
            ceil(float(params.one_core_data * params.updates_data_num) / params.block_number);
        params.tail_updates_burst_len =
            ceil(float(params.last_core_data_num * params.updates_data_num) / params.block_number);
        if (params.last_core_data_num > MAX_UB_CORE_INDICES) {
          const uint32_t& tail_indices_num = params.last_core_data_num % MAX_UB_CORE_INDICES;
          params.tail_indices_num_burst_len = ceil(float(tail_indices_num) / params.block_number);
        }
      } else {
        // deal data more than 474
        params.select_mode = TILING_MODE_1;
        params.select_params = SELECT_MORE_THAN_PARAMS;
        const uint32_t& tail_indices_num = params.one_core_data % MAX_UB_CORE_INDICES;
        params.tail_indices_burst_len = ceil(float(tail_indices_num) / params.block_number);
        if (params.last_core_data_num > MAX_UB_CORE_INDICES) {
          const uint32_t& tail_last_indices_number = params.last_core_data_num % MAX_UB_CORE_INDICES;
          params.tail_indices_more_than_burst_len = ceil(float(tail_last_indices_number) / params.block_number);
        }
      }
    }
  }

  SetRunningParams(params, runInfo);

  runInfo.block_dim = params.core_num;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(ScatterNd, ScatterNdTiling);
}  // namespace optiling
