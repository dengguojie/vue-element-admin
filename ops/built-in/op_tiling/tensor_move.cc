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
 * \file tensor_move.cc
 * \brief
 */
#include <math.h>

#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace optiling {

const int64_t TILING_MODE_1 = 1;

struct TensorMoveTilingParam {
  int64_t tilingMode;
  int64_t dataSize;
  int64_t needCoreNum;
};

void InitRunningParams(TensorMoveTilingParam& params) {
  params.tilingMode = TILING_MODE_1;
  params.dataSize = 0;
  params.needCoreNum = 0;
}

void SetRunningParam(const TensorMoveTilingParam& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, params.dataSize);
  ByteBufferPut(runInfo.tiling_data, params.needCoreNum);
}

void PrintTilingParams(const std::string& opType, const TensorMoveTilingParam& params) {
  OP_LOGD(opType.c_str(), "op [TensorMoveTiling] : tilingMode=%d.", params.tilingMode);
  OP_LOGD(opType.c_str(), "op [TensorMoveTiling] : dataSize=%d.", params.dataSize);
  OP_LOGD(opType.c_str(), "op [TensorMoveTiling] : needCoreNum=%d.", params.needCoreNum);
}

bool GetTensorMoveCompileParams(const std::string& opType,
                                const nlohmann::json& opCompileInfo,
                                int64_t& coreNum,
                                int64_t& dataLenOneBlock) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfo["vars"];

  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get core_num error.");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();

  if (allVars.count("data_len_one_block") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get data_len_one_block error.");
    return false;
  }
  dataLenOneBlock = allVars["data_len_one_block"].get<std::int64_t>();

  return true;
}

int64_t get_ceil_int(int64_t dividend, int64_t divisor) {
  int64_t offset;
  if (divisor == 0) {
    OP_LOGD("Divisor cannot be 0.");
  }
  offset = (dividend + divisor - 1) / divisor;
  return offset;
}

bool TensorMoveTiling(const std::string& opType, const TeOpParas& opParas,
                      const nlohmann::json& opCompileInfo, OpRunInfo& runInfo) {
  using namespace ge;

  OP_LOGI(opType.c_str(), "TensorMove running.");
  if (opCompileInfo == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "opCompileInfo json error.");
    return false;
  }

  if (opParas.inputs.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "input shape error.");
    return false;
  }

  if (opParas.outputs.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "output shape error.");
    return false;
  }

  int64_t coreData = 0;
  int64_t coreNum = 0;
  int64_t coreUsed = 0;
  int64_t dataLenOneBlock = 0;
  int64_t shapeListNum = 1;

  const std::vector<int64_t>& inputShape = opParas.inputs[0].tensor[0].shape;
  std::string inputDtype = opParas.inputs[0].tensor[0].dtype;

  bool can_get_params = GetTensorMoveCompileParams(opType, opCompileInfo, coreNum, dataLenOneBlock);
  if (!can_get_params) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetTensorMoveCompileParams errors.");
    return false;
  }

  for (int64_t i = 0; i < inputShape.size(); i++) {
    shapeListNum *= inputShape[i];
  }

  coreData = get_ceil_int(shapeListNum, coreNum);
  coreData = get_ceil_int(coreData, dataLenOneBlock) * dataLenOneBlock;
  coreUsed = get_ceil_int(shapeListNum, coreData);

  TensorMoveTilingParam runParams;
  InitRunningParams(runParams);

  runParams.tilingMode = TILING_MODE_1;
  runParams.dataSize = shapeListNum;
  runParams.needCoreNum = coreUsed;

  SetRunningParam(runParams, runInfo);
  PrintTilingParams(opType, runParams);

  runInfo.block_dim = runParams.needCoreNum;
  OP_LOGI(opType.c_str(), "TensorMoveTiling run success.");

  return true;
}

// register tiling interface of the TensorMove op.
REGISTER_OP_TILING_FUNC_BUFFERED(TensorMove, TensorMoveTiling);
}  // namespace optiling
