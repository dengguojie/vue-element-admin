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
 * \file scatter_add.cpp
 * \brief tiling function of scatteradd
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "register/op_tiling.h"

#include "../op_proto/util/error_util.h"
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
 * @brief: tiling function of scatteradd
 * @param [in] opType: opType of the scatteradd
 * @param [in] opParas: inputs/outputs/atts of the scatteradd
 * @param [in] opCompileInfo: compile time generated info of the scatteradd
 * @param [out] runInfo: result data
 * @return bool: success or not
 */

const uint32_t BYTE_BLOCK = 32;
typedef enum { FP16_BYTE = 2, FP32_BYTE = 4, INT32_BYTE = 4 } EleByte;

struct ScatterAdd {
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

bool CheckScatterAddTensorShape(const std::string& opType, std::vector<int64_t> varShape,
                                std::vector<int64_t> indicesShape, std::vector<int64_t> updateShape,
                                std::vector<int64_t> outputShape) {
  if (indicesShape.size() == 0) {
    ge::OpsOneInputShapeErrReport("ScatterAdd", "indices", "the size of indices's shape is empty");
    OP_LOGE("op ScatterAddTiling : the input of indices's shape is empty");
    return false;
  }
  const uint32_t& varDims = varShape.size();
  const uint32_t& indicesDims = indicesShape.size();
  const uint32_t& outputDims = outputShape.size();
  const uint32_t& updateDims = updateShape.size();
  if (varDims != outputDims) {
    ge::OpsOneInputShapeErrReport("ScatterAdd", "var", "the length of var must be same as the length of output");
    OP_LOGE(opType.c_str(), "op ScatterAddTiling : the length of var must be same as the length of output");
    return false;
  }
  for (size_t i = 0; i < varDims; i++) {
    if (varShape[i] != outputShape[i]) {
      ge::OpsOneOutputShapeErrReport("ScatterAdd", "var_out", "the var's shape is not equal to var_out's shape");
      OP_LOGE(opType.c_str(), "op ScatterAddTiling : the output of shape is invalid");
      return false;
    }
  }
  if (indicesShape[indicesDims - 1] == 0 && indicesDims == 1) {
    ge::OpsOneInputShapeErrReport("ScatterAdd", "indices",
                                  "the length of indices is one and the last dim of indices's shape is zero");
    OP_LOGE(opType.c_str(), "op ScatterAddtiling : the indices of shape is invalid");
    return false;
  }
  for (size_t i = 0; i < indicesDims; i++) {
    if (indicesShape[i] != updateShape[i]) {
      ge::OpsOneInputShapeErrReport("ScatterAdd", "update", "the indices's shape is not equal to update's shape");
      OP_LOGE(opType.c_str(), "op ScatterAddTiling : the input update is invalid");
      return false;
    }
  }
  if (indicesDims + varDims - 1 != updateDims) {
    ge::OpsOneInputShapeErrReport("ScatterAdd", "update", "update does not satisfy the relation expression with var");
    OP_LOGE(opType.c_str(), "op ScatterAddTiling : the input update is invalid");
    return false;
  }
  for (size_t i = 1; i < varDims; i++) {
    if (varShape[i] != updateShape[indicesDims + i - 1]) {
      ge::OpsOneInputShapeErrReport("ScatterAdd", "update", "update does not satisfy the relation expression with var");
      OP_LOGE(opType.c_str(), "op ScatterAddTiling : the input update is invalid");
      return false;
    }
  }

  return true;
}

bool GetSocParamsOne(const std::string& opType, const nlohmann::json& opCompileInfo, uint32_t& coreNum,
                     uint32_t& ubSize) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "core_num");
    OP_LOGE(opType.c_str(), "op ScatterAddTiling : GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::uint32_t>();
  if (allVars.count("ub_size") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "ub_size");
    OP_LOGE(opType.c_str(), "op ScatterAddTiling : GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::uint32_t>();
  return true;
}

void tiling_indices_core(uint32_t& coreNum, uint32_t& inputSize, uint32_t& one_core_data,
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

void InitRunningParams(ScatterAdd& params) {
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

void SetRuningParams(const ScatterAdd& params, OpRunInfo& runInfo) {
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

bool ScatterAddTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                      OpRunInfo& runInfo) {
  using namespace nlohmann;
  OP_LOGI(opType.c_str(), "op ScatterAddTiling running.");
  if (opCompileInfo == nullptr) {
    OP_LOGE(opType.c_str(), "op ScatterAddTiling: opCompileInfo json error.");
    return false;
  }
  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty() || opParas.inputs[1].tensor.empty() ||
      opParas.inputs[2].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices or updates or var", "the input may be empty");
    OP_LOGE(opType.c_str(), "op ScatterAddTiling: input shape error");
    return false;
  }
  if (opParas.outputs.empty() || opParas.outputs[0].tensor.empty()) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "var_out", "the output may be empty");
    OP_LOGE(opType.c_str(), "op ScatterAddTiling: output shape error");
    return false;
  }
  const std::vector<int64_t>& inputShape0 = opParas.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& inputShape1 = opParas.inputs[1].tensor[0].shape;
  const std::vector<int64_t>& inputShape2 = opParas.inputs[2].tensor[0].shape;
  const std::vector<int64_t>& outputShape = opParas.outputs[0].tensor[0].shape;
  if (inputShape0.size() < 2) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "var", "the input may be empty");
    OP_LOGE(opType.c_str(), "op ScatterAddTiling: input shape error");
    return false;
  }
  if (outputShape.size() < 2) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "var_out", "the output may be empty");
    OP_LOGE(opType.c_str(), "op ScatterAddTiling: output shape error");
    return false;
  }

  bool ret = CheckScatterAddTensorShape(opType, inputShape0, inputShape1, inputShape2, outputShape);
  if (!ret) {
    OP_LOGE(opType.c_str(), "op ScatterAddTiling : checktensorshape is failed");
    return false;
  }
  ScatterAdd params;
  InitRunningParams(params);

  params.indices_ub_number = std::accumulate(inputShape1.begin(), inputShape1.end(), 1, std::multiplies<int>());
  const int32_t& updatesize = std::accumulate(inputShape2.begin(), inputShape2.end(), 1, std::multiplies<int>());

  const std::string& update_dtype = opParas.inputs[2].tensor[0].dtype;
  EleByte input_update_byte;
  if (update_dtype == "float32" || update_dtype == "int32") {
    input_update_byte = FP32_BYTE;
  } else {
    input_update_byte = FP16_BYTE;
  }
  params.block_number = BYTE_BLOCK / input_update_byte;
  uint32_t ubSize = 0;
  uint32_t coreNum = 0;
  bool flag = GetSocParamsOne(opType, opCompileInfo, coreNum, ubSize);
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
    tiling_indices_core(coreNum, params.indices_ub_number, final_one_core_data_num_one, last_core_data_num_one);
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

  runInfo.block_dim = params.core_num;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  OP_LOGI(opType.c_str(), "op tiling run success.");
  return true;
}

REGISTER_OP_TILING_FUNC(ScatterAdd, ScatterAddTiling);
}  // namespace optiling
