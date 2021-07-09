/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file inplace_index_add.cpp
 * \brief
 */
#include <math.h>

#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"


namespace optiling {

const int32_t BLOCK_SIZE = 32;
const int32_t TILING_MODE_1 = 1;
const int32_t TILING_MODE_2 = 2;
const int32_t TILING_MODE_3 = 3;
const int32_t TILING_MODE_4 = 4;
const int32_t TILING_MODE_5 = 5;
const int32_t TILING_MODE_6 = 6;
const int32_t TILING_MODE_7 = 7;

struct InplaceIndexAddTilingParam {
  int32_t tilingMode;
  int32_t blockNum;
  int32_t indicesNum;
  int32_t outerLoop;
  int32_t outerLoopPerBlock;
  int32_t axisAndAfterDataNumOfUpdates;
  int32_t axisAndAfterDataNumOfVar;
  int32_t updateDataNum;
};

void InitRunningParams(InplaceIndexAddTilingParam& params) {
  params.tilingMode = TILING_MODE_1;
  params.blockNum = 0;
  params.indicesNum = 0;
  params.outerLoop = 1;
  params.outerLoopPerBlock = 0;
  params.axisAndAfterDataNumOfUpdates = 1;
  params.axisAndAfterDataNumOfVar = 1;
  params.updateDataNum = 1;
}

void SetRunningParam(const InplaceIndexAddTilingParam& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, params.blockNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesNum);
  ByteBufferPut(runInfo.tiling_data, params.outerLoop);
  ByteBufferPut(runInfo.tiling_data, params.outerLoopPerBlock);
  ByteBufferPut(runInfo.tiling_data, params.axisAndAfterDataNumOfUpdates);
  ByteBufferPut(runInfo.tiling_data, params.axisAndAfterDataNumOfVar);
  ByteBufferPut(runInfo.tiling_data, params.updateDataNum);
}

void PrintTilingParams(const std::string& opType, const InplaceIndexAddTilingParam& params) {
  OP_LOGD(opType.c_str(), "op [InplaceIndexAddTiling] : tilingMode=%ld.", params.tilingMode);
  OP_LOGD(opType.c_str(), "op [InplaceIndexAddTiling] : blockNum=%ld.", params.blockNum);
  OP_LOGD(opType.c_str(), "op [InplaceIndexAddTiling] : indicesNum=%ld.", params.indicesNum);
  OP_LOGD(opType.c_str(), "op [InplaceIndexAddTiling] : outerLoop=%ld.", params.outerLoop);
  OP_LOGD(opType.c_str(), "op [InplaceIndexAddTiling] : outerLoopPerBlock=%ld.", params.outerLoopPerBlock);
  OP_LOGD(opType.c_str(), "op [InplaceIndexAddTiling] : axisAndAfterDataNumOfUpdates=%ld.", params.axisAndAfterDataNumOfUpdates);
  OP_LOGD(opType.c_str(), "op [InplaceIndexAddTiling] : axisAndAfterDataNumOfVar=%ld.", params.axisAndAfterDataNumOfVar);
  OP_LOGD(opType.c_str(), "op [InplaceIndexAddTiling] : updateDataNum=%ld.", params.updateDataNum);
}

bool GetInplaceIndexAddCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int32_t& coreNum,
                                     int32_t& ubSize, int32_t& varSize, int32_t& indicesSize, int32_t& vconvSize, int32_t& axis) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfo["vars"];

  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get core_num error.");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int32_t>();

  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get ub_size error.");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int32_t>();

  if (allVars.count("var_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get var_size error.");
    return false;
  }
  varSize = allVars["var_size"].get<std::int32_t>();

  if (allVars.count("indices_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get indices_size error.");
    return false;
  }
  indicesSize = allVars["indices_size"].get<std::int32_t>();

  if (allVars.count("vconv_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get vconv_size error.");
    return false;
  }
  indicesSize = allVars["vconv_size"].get<std::int32_t>();

  if (allVars.count("axis") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get axis error.");
    return false;
  }
  axis = allVars["axis"].get<std::int32_t>();

  return true;
}

bool InplaceIndexAddTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo, OpRunInfo& runInfo) {
  using namespace ge;

  OP_LOGI(opType.c_str(), "InplaceIndexAdd running.");
  if (opCompileInfo == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "opCompileInfo json error.");
    return false;
  }

  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty() || opParas.inputs[1].tensor.empty() || opParas.inputs[2].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "input shape error.");
    return false;
  }

  if (opParas.outputs.empty() || opParas.outputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "output shape error.");
    return false;
  }

  const std::vector<int64_t>& varShape = opParas.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& indicesShape = opParas.inputs[1].tensor[0].shape;
  const std::vector<int64_t>& updatesShape = opParas.inputs[2].tensor[0].shape;
  std::string inputDtype = opParas.inputs[0].tensor[0].dtype;

  int32_t coreNum = 0;
  int32_t ubSize = 0;
  int32_t varSize = 0;
  int32_t indicesSize = 0;
  int32_t vconvSize = 0;
  int32_t axis = 0;
  bool can_get_params = GetInplaceIndexAddCompileParams(opType, opCompileInfo, coreNum, ubSize, varSize, indicesSize, vconvSize, axis);
  if (!can_get_params) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetInplaceIndexAddCompileParams errors.");
    return false;
  }

  InplaceIndexAddTilingParam runParams;
  InitRunningParams(runParams);
  runParams.indicesNum = std::accumulate(indicesShape.begin(), indicesShape.end(), 1, std::multiplies<int>());

  for (int32_t i = 0; i < axis; i++) {
    runParams.outerLoop = runParams.outerLoop * varShape[i];
  }

  if (runParams.outerLoop == 1) {
    runParams.blockNum = 1;
    runParams.outerLoopPerBlock = 1;
  } else {
    runParams.outerLoopPerBlock = (runParams.outerLoop - 1) / coreNum + 1;
    runParams.blockNum = (runParams.outerLoop - 1) / runParams.outerLoopPerBlock + 1;
  }

  for (size_t i = axis; i < updatesShape.size(); i++) {
    runParams.axisAndAfterDataNumOfUpdates = runParams.axisAndAfterDataNumOfUpdates * updatesShape[i];
  }
  for (size_t i = axis; i < varShape.size(); i++) {
    runParams.axisAndAfterDataNumOfVar = runParams.axisAndAfterDataNumOfVar * varShape[i];
  }
  for (size_t i = axis+1; i < varShape.size(); i++) {
    runParams.updateDataNum = runParams.updateDataNum * varShape[i];
  }

  int32_t updatesSizeBytes = varSize * runParams.updateDataNum;
  int32_t indicesSizeBytes = indicesSize * runParams.indicesNum;
  int32_t vconvSizeBtytes = runParams.updateDataNum * vconvSize;

  if (inputDtype == "int8_t" || inputDtype == "uint8_t") {
    if ((updatesSizeBytes + vconvSizeBtytes) * 2 < ubSize) {
      runParams.tilingMode = TILING_MODE_4;
    } else if (indicesSizeBytes < ubSize) {
      runParams.tilingMode = TILING_MODE_5;
    } else {
      runParams.tilingMode = TILING_MODE_6;
    }
  } else {
    if (updatesSizeBytes * 2 < ubSize) {
      runParams.tilingMode = TILING_MODE_1;
    } else if (indicesSize < ubSize) {
      runParams.tilingMode = TILING_MODE_2;
    } else {
      runParams.tilingMode = TILING_MODE_3;
    }
  }

  if (updatesSizeBytes <= 32) {
    runParams.tilingMode = TILING_MODE_7;
    runParams.blockNum = 1;
    runParams.outerLoopPerBlock = runParams.outerLoop;
  }

  SetRunningParam(runParams, runInfo);

  PrintTilingParams(opType, runParams);

  runInfo.block_dim = runParams.blockNum;

  OP_LOGI(opType.c_str(), "InplaceIndexAddTiling run success.");

  return true;
}

// register tiling interface of the InplaceIndexAdd op.
REGISTER_OP_TILING_FUNC_BUFFERED(InplaceIndexAdd, InplaceIndexAddTiling);
}  // namespace optiling
