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
 * \brief
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

const int64_t BLOCK_SIZE = 32;
const int64_t OUT_SPECIAL_DIM_0 = 640000;
const int64_t OUT_SPECIAL_DIM_1 = 1;
const int64_t OUT_SPECIAL_DIM_2 = 80;
const int64_t OUT_SPECIAL_DIM_3 = 300000;
const int64_t OUT_SPECIAL_DIM_4 = 256;
const int64_t OUT_SPECIAL_DIM_5 = 279424;
// 32b aligned, ub can store all updatesNum, float32 atomic
const int64_t TILING_MODE_1 = 1;
// 32b aligned, ub can't store all updatesNum, float32 atomic
const int64_t TILING_MODE_2 = 2;
// updateDataNum is less than 1 block, ub can store all updatesNum, float32 atomic
const int64_t TILING_MODE_3 = 3;
// updateDataNum is less than 1 block, ub can't store all updatesNum, float32 atomic
const int64_t TILING_MODE_4 = 4;
// updateDataNum is more than 1 block, float32 atomic
const int64_t TILING_MODE_5 = 5;
// 32b aligned, ub can store all var and updates, not atomic
const int64_t TILING_MODE_6 = 6;
// 32b aligned, ub can store all var, not atomic
const int64_t TILING_MODE_7 = 7;
// 32b aligned, ub can store all updates, not atomic
const int64_t TILING_MODE_8 = 8;
// 32b aligned, ub can't store all var and updates, not atomic
const int64_t TILING_MODE_9 = 9;
// updateDataNum is less than 1 block, ub can store all var and updates, not atomic
const int64_t TILING_MODE_10 = 10;
// updateDataNum is less than 1 block, ub can store all var, not atomic
const int64_t TILING_MODE_11 = 11;
// updateDataNum is less than 1 block, ub can store all updates, not atomic
const int64_t TILING_MODE_12 = 12;
// updateDataNum is less than 1 block, ub can't store all var and updates, not atomic
const int64_t TILING_MODE_13 = 13;
// updateDataNum is more than 1 block, not atomic, and less than updateubnum
const int64_t TILING_MODE_14 = 14;
// updateDataNum is more than 1 block, not atomic, and more than updateubnum
const int64_t TILING_MODE_15 = 15;
// high perf branch, updateDataNum is less than 1 block
const int64_t TILING_MODE_16 = 16;
// high perf branch, updateDataNum is 32b aligned
const int64_t TILING_MODE_17 = 17;
// div 0 check
const int64_t ZERO = 0;

struct ScatterNdTilingParams {
  int64_t tilingMode;
  int64_t indiceStep;
  int64_t coreNum;
  int64_t updatesDataNum;
  int64_t indicesLoopNum;
  int64_t indicesLastNum;
  int64_t updatesNum;
  int64_t updatesLoopNum;
  int64_t updatesLastNum;
  int64_t varNum;
  int64_t varLoopNum;
  int64_t varLastNum;
  int64_t varEachCoreBurstLen;
  int64_t varLastCoreBurstLen;
  int64_t maxIndice;
  int64_t varEachCoreData;
  int64_t indicesLastDim;
  std::vector<int64_t> varOffSet = {0, 0, 0, 0, 0, 0, 0};
  int64_t varEachCoreSetZeroLoopNum;
  int64_t varEachCoreSetZeroLastNum;
  int64_t varLastCoreSetZeroLoopNum;
  int64_t varLastCoreSetZeroLastNum;
  int64_t indicesEachCoreData;
  int64_t indicesLastCoreData;
  int64_t eachCoreIndicesLoopNum;
  int64_t eachCoreIndicesLastNum;
  int64_t lastCoreIndicesLoopNum;
  int64_t lastCoreIndicesLastNum;
};

void InitRunningParams(ScatterNdTilingParams& params) {
  params.tilingMode = TILING_MODE_1;
  params.indiceStep = 0;
  params.coreNum = 0;
  params.updatesDataNum = 0;
  params.indicesLoopNum = 0;
  params.indicesLastNum = 0;
  params.updatesNum = 0;
  params.updatesLoopNum = 0;
  params.updatesLastNum = 0;
  params.varNum = 0;
  params.varLoopNum = 0;
  params.varLastNum = 0;
  params.varEachCoreBurstLen = 0;
  params.varLastCoreBurstLen = 0;
  params.maxIndice = 0;
  params.varEachCoreData = 0;
  params.indicesLastDim = 0;
  params.varEachCoreSetZeroLoopNum = 0;
  params.varEachCoreSetZeroLastNum = 0;
  params.varLastCoreSetZeroLoopNum = 0;
  params.varLastCoreSetZeroLastNum = 0;
  params.indicesEachCoreData = 0;
  params.indicesLastCoreData = 0;
  params.eachCoreIndicesLoopNum = 0;
  params.eachCoreIndicesLastNum = 0;
  params.lastCoreIndicesLoopNum = 0;
  params.lastCoreIndicesLastNum = 0;
}

void CalAtomicBranchRunningParams(ScatterNdTilingParams& runParams, int64_t indicesNum, int64_t updatesNum,
                                  int64_t updateDataNum, int64_t ubSize, int64_t updatesSize, int64_t indicesSize,
                                  int64_t updatesDataEachBlock) {
  int64_t updateSizeByte = updatesSize * updatesNum;
  int64_t halfUbSize = ubSize / 2;
  runParams.updatesLoopNum = updateDataNum / (halfUbSize / updatesSize);
  runParams.updatesLastNum = updateDataNum % (halfUbSize / updatesSize);
  runParams.indicesLoopNum =
      indicesNum / (halfUbSize / indicesSize / runParams.indicesLastDim * runParams.indicesLastDim);
  runParams.indicesLastNum =
      indicesNum % (halfUbSize / indicesSize / runParams.indicesLastDim * runParams.indicesLastDim);
  runParams.updatesDataNum = updateDataNum;
  runParams.updatesNum = updatesNum;

  if (updateDataNum % updatesDataEachBlock == 0) {
    if (updateSizeByte <= halfUbSize) {
      runParams.tilingMode = TILING_MODE_1;
    } else {
      runParams.tilingMode = TILING_MODE_2;
    }
  } else {
    if (updateDataNum < updatesDataEachBlock) {
      if (updateSizeByte <= halfUbSize) {
        runParams.tilingMode = TILING_MODE_3;
        runParams.updatesLoopNum = updatesNum / (halfUbSize / updatesSize);
        runParams.updatesLastNum = updatesNum % (halfUbSize / updatesSize);
      } else {
        runParams.tilingMode = TILING_MODE_4;
      }
    } else {
      runParams.tilingMode = TILING_MODE_5;
    }
  }
}

void CalNotAtomicBranchRunningParams(ScatterNdTilingParams& runParams, int64_t varNum, int64_t indicesNum,
                                     int64_t updatesNum, int64_t updateDataNum, int64_t maxIndice, int64_t ubSize,
                                     int64_t coreNum, int64_t updatesSize, int64_t indicesSize,
                                     int64_t updatesDataEachBlock, int64_t dataNumOneRepeat) {
  int64_t varAllSizeByte = updatesSize * varNum;
  int64_t varSizeByte = updatesSize * runParams.indiceStep * updateDataNum;
  int64_t updateSizeByte = updatesSize * updatesNum;
  int64_t varUbSize = ubSize / 8 * 3;
  int64_t indicesUbSize = ubSize / 8 * 2;
  runParams.varLoopNum = varNum / (varUbSize / updatesSize);
  runParams.varLastNum = varNum % (varUbSize / updatesSize);
  runParams.updatesLoopNum = updateDataNum / (varUbSize / updatesSize);
  runParams.updatesLastNum = updateDataNum % (varUbSize / updatesSize);
  runParams.indicesLoopNum =
      indicesNum / (indicesUbSize / indicesSize / runParams.indicesLastDim * runParams.indicesLastDim);
  runParams.indicesLastNum =
      indicesNum % (indicesUbSize / indicesSize / runParams.indicesLastDim * runParams.indicesLastDim);
  runParams.updatesDataNum = updateDataNum;
  runParams.updatesNum = updatesNum;
  runParams.varNum = varNum;
  if (updateDataNum % updatesDataEachBlock == 0) {
    if (updateSizeByte <= varUbSize && varSizeByte <= varUbSize) {
      runParams.tilingMode = TILING_MODE_6;
    } else if (updateSizeByte > varUbSize && varSizeByte <= varUbSize) {
      runParams.tilingMode = TILING_MODE_7;
    } else if (updateSizeByte <= varUbSize && varSizeByte > varUbSize) {
      runParams.tilingMode = TILING_MODE_8;
    } else {
      runParams.tilingMode = TILING_MODE_9;
    }
  } else if (updateDataNum < updatesDataEachBlock) {
    if (updateSizeByte <= varUbSize && varAllSizeByte <= varUbSize) {
      runParams.tilingMode = TILING_MODE_10;
    } else if (updateSizeByte > varUbSize && varAllSizeByte <= varUbSize) {
      runParams.tilingMode = TILING_MODE_11;
    } else if (updateSizeByte <= varUbSize && varAllSizeByte > varUbSize) {
      runParams.tilingMode = TILING_MODE_12;
    } else {
      runParams.tilingMode = TILING_MODE_13;
    }
  } else {
    if (updateDataNum / (varUbSize / updatesSize) == 0) {
      runParams.tilingMode = TILING_MODE_14;
    } else {
      runParams.tilingMode = TILING_MODE_15;
    }
  }

  runParams.varEachCoreData = runParams.indiceStep * runParams.updatesDataNum;
  int64_t varLastCoreData = varNum - runParams.varEachCoreData * (coreNum - 1);
  runParams.varEachCoreBurstLen = runParams.varEachCoreData / updatesDataEachBlock;
  runParams.varLastCoreBurstLen = varLastCoreData / updatesDataEachBlock;

  runParams.varEachCoreSetZeroLoopNum = runParams.varEachCoreData / (varUbSize / updatesSize);
  runParams.varEachCoreSetZeroLastNum = runParams.varEachCoreData % (varUbSize / updatesSize);
  runParams.varLastCoreSetZeroLoopNum = varLastCoreData / (varUbSize / updatesSize);
  runParams.varLastCoreSetZeroLastNum = varLastCoreData % (varUbSize / updatesSize);

  if (runParams.tilingMode == TILING_MODE_9 || runParams.tilingMode == TILING_MODE_14 ||
      runParams.tilingMode == TILING_MODE_15) {
    runParams.varLoopNum = updateDataNum / (varUbSize / updatesSize);
    runParams.varLastNum = updateDataNum % (varUbSize / updatesSize);
  }
}

void CalScatterNdHighPerfBranchParams(ScatterNdTilingParams& runParams, int64_t indicesNum, int64_t coreNum,
                                      int64_t ubSize, int64_t updateDataNum, int64_t updatesDataEachBlock,
                                      int64_t indicesSize) {
  int64_t halfUbSize = ubSize / 2;
  runParams.tilingMode = TILING_MODE_16;
  runParams.updatesDataNum = updateDataNum;
  runParams.indicesEachCoreData = ceil(float(indicesNum) / coreNum);
  runParams.coreNum = ceil(float(indicesNum) / runParams.indicesEachCoreData);
  runParams.indicesLastCoreData = indicesNum - runParams.indicesEachCoreData * (runParams.coreNum - 1);
  runParams.eachCoreIndicesLoopNum = runParams.indicesEachCoreData / (halfUbSize / indicesSize);
  runParams.eachCoreIndicesLastNum = runParams.indicesEachCoreData % (halfUbSize / indicesSize);
  runParams.lastCoreIndicesLoopNum = runParams.indicesLastCoreData / (halfUbSize / indicesSize);
  runParams.lastCoreIndicesLastNum = runParams.indicesLastCoreData % (halfUbSize / indicesSize);

  if (updateDataNum % updatesDataEachBlock == 0) {
    runParams.tilingMode = TILING_MODE_17;
  }
}

void SetRuningParams(const ScatterNdTilingParams& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, params.indiceStep);
  ByteBufferPut(runInfo.tiling_data, params.coreNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesDataNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLastNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesLastNum);
  ByteBufferPut(runInfo.tiling_data, params.varNum);
  ByteBufferPut(runInfo.tiling_data, params.varLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.varLastNum);
  ByteBufferPut(runInfo.tiling_data, params.varEachCoreBurstLen);
  ByteBufferPut(runInfo.tiling_data, params.varLastCoreBurstLen);
  ByteBufferPut(runInfo.tiling_data, params.maxIndice);
  ByteBufferPut(runInfo.tiling_data, params.varEachCoreData);
  ByteBufferPut(runInfo.tiling_data, params.indicesLastDim);
  for (size_t i = 0; i < params.varOffSet.size(); i++) {
    ByteBufferPut(runInfo.tiling_data, params.varOffSet[i]);
  }
  ByteBufferPut(runInfo.tiling_data, params.varEachCoreSetZeroLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.varEachCoreSetZeroLastNum);
  ByteBufferPut(runInfo.tiling_data, params.varLastCoreSetZeroLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.varLastCoreSetZeroLastNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesEachCoreData);
  ByteBufferPut(runInfo.tiling_data, params.indicesLastCoreData);
  ByteBufferPut(runInfo.tiling_data, params.eachCoreIndicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.eachCoreIndicesLastNum);
  ByteBufferPut(runInfo.tiling_data, params.lastCoreIndicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.lastCoreIndicesLastNum);
}

void PrintTilingParams(const std::string& opType, const ScatterNdTilingParams& params) {
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : tilingMode=%ld.", params.tilingMode);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : indiceStep=%ld.", params.indiceStep);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : coreNum=%ld.", params.coreNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : updatesDataNum=%ld.", params.updatesDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : indicesLoopNum=%ld.", params.indicesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : indicesLastNum=%ld.", params.indicesLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : updatesNum=%ld.", params.updatesNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : updatesLoopNum=%ld.", params.updatesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : updatesLastNum=%ld.", params.updatesLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varNum=%ld.", params.varNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varLoopNum=%ld.", params.varLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varLastNum=%ld.", params.varLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varEachCoreBurstLen=%ld.", params.varEachCoreBurstLen);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varLastCoreBurstLen=%ld.", params.varLastCoreBurstLen);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : maxIndice=%ld.", params.maxIndice);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varEachCoreData=%ld.", params.varEachCoreData);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : indicesLastDim=%ld.", params.indicesLastDim);
  for (size_t i = 0; i < params.varOffSet.size(); i++) {
    OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varOffSet[%ld]=%ld.", i, params.varOffSet[i]);
  }
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varEachCoreSetZeroLoopNum=%ld.", params.varEachCoreSetZeroLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varEachCoreSetZeroLastNum=%ld.", params.varEachCoreSetZeroLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varLastCoreSetZeroLoopNum=%ld.", params.varLastCoreSetZeroLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varLastCoreSetZeroLastNum=%ld.", params.varLastCoreSetZeroLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : indicesEachCoreData=%ld.", params.indicesEachCoreData);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : indicesLastCoreData=%ld.", params.indicesLastCoreData);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : eachCoreIndicesLoopNum=%ld.", params.eachCoreIndicesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : eachCoreIndicesLastNum=%ld.", params.eachCoreIndicesLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : lastCoreIndicesLoopNum=%ld.", params.lastCoreIndicesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : lastCoreIndicesLastNum=%ld.", params.lastCoreIndicesLastNum);
}

bool CheckScatterNdTensorShape(const std::string& opType, std::vector<int64_t> indicesShape,
                               std::vector<int64_t> updatesShape, std::vector<int64_t> outputShape) {
  const int64_t& indicesDims = indicesShape.size();
  const int64_t& updatesDims = updatesShape.size();
  const int64_t& outputDims = outputShape.size();
  int64_t indicesLastDim = indicesShape.back();

  if (indicesDims <= 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "the ndim of indices is less than 1 or equal to 1");
    return false;
  }

  if (indicesDims - 1 + outputDims - indicesLastDim != updatesDims) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "output's shape and updates'shape are not equal in some dimensions");
    return false;
  }

  for (int64_t i = 0; i < indicesDims - 1; i++) {
    if (indicesShape[i] != updatesShape[i]) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "indices's shape and updates'shape are not equal in some dimensions");
      return false;
    }
  }

  for (int64_t i = 0; i < updatesDims - indicesDims + 1; i++) {
    if (updatesShape[indicesDims - 1 + i] != outputShape[indicesLastDim + i]) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "output's shape and updates'shape are not equal in some dimensions");
      return false;
    }
  }
  return true;
}

bool GetScatterNdCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int64_t& coreNum,
                               int64_t& ubSize, int64_t& updatesSize, int64_t& indicesSize, int64_t& supportAtomic) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();

  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int64_t>();

  if (allVars.count("updates_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get updates_size error");
    return false;
  }
  updatesSize = allVars["updates_size"].get<std::int64_t>();

  if (allVars.count("indices_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get indices_size error");
    return false;
  }
  indicesSize = allVars["indices_size"].get<std::int64_t>();

  if (allVars.count("support_atomic") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get support_atomic error");
    return false;
  }
  supportAtomic = allVars["support_atomic"].get<std::int64_t>();

  return true;
}

bool CheckScatterNdHighPerfShape(std::vector<int64_t> outShape, std::vector<int64_t> indicesShape) {
  if (indicesShape.size() != 2 || outShape.size() != 2 || indicesShape[1] != 1) {
    return false;
  }

  if ((outShape[0] == OUT_SPECIAL_DIM_0 && outShape[1] == OUT_SPECIAL_DIM_1) ||
      (outShape[0] == OUT_SPECIAL_DIM_0 && outShape[1] == OUT_SPECIAL_DIM_2) ||
      (outShape[0] == OUT_SPECIAL_DIM_3 && outShape[1] == OUT_SPECIAL_DIM_4) ||
      (outShape[0] == OUT_SPECIAL_DIM_5 && outShape[1] == OUT_SPECIAL_DIM_1)) {
    return true;
  }
  return false;
}

bool ScatterNdTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                     OpRunInfo& runInfo) {
  using namespace ge;

  OP_LOGI(opType.c_str(), "ScatterNdTiling running.");
  if (opCompileInfo == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "opCompileInfo json error.");
    return false;
  }

  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty() || opParas.inputs[1].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "input shape error");
    return false;
  }

  if (opParas.outputs.empty() || opParas.outputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "output shape error");
    return false;
  }

  if (opParas.const_inputs.find("shape") == opParas.const_inputs.end()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "get const input failed.");
    return false;
  }

  const std::vector<int64_t>& indicesShape = opParas.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& updatesShape = opParas.inputs[1].tensor[0].shape;
  const std::vector<int64_t>& outShape = opParas.outputs[0].tensor[0].shape;
  std::string input_dtype = opParas.inputs[1].tensor[0].dtype;

  bool is_valid_shape = CheckScatterNdTensorShape(opType, indicesShape, updatesShape, outShape);
  if (!is_valid_shape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckScatterNdTensorShape is failed");
    return false;
  }

  int64_t coreNum = 0;
  int64_t ubSize = 0;
  int64_t updatesSize = 0;
  int64_t indicesSize = 0;
  int64_t supportAtomic = 0;
  bool can_get_params =
      GetScatterNdCompileParams(opType, opCompileInfo, coreNum, ubSize, updatesSize, indicesSize, supportAtomic);
  if (!can_get_params) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetScatterNdCompileParams error");
    return false;
  }
  if (coreNum <= ZERO || ubSize <= ZERO || updatesSize <= ZERO || indicesSize <= ZERO) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        opType, "coreNum, ubSize, updatesSize, indicesSize must be greater to 0, but got %ld, %ld, %ld, %ld", coreNum, ubSize,
        updatesSize, indicesSize);
    return false;
  }

  ScatterNdTilingParams runParams;
  InitRunningParams(runParams);

  int64_t varNum = std::accumulate(outShape.begin(), outShape.end(), 1, std::multiplies<int>());
  int64_t indicesNum = std::accumulate(indicesShape.begin(), indicesShape.end(), 1, std::multiplies<int>());
  int64_t updatesNum = std::accumulate(updatesShape.begin(), updatesShape.end(), 1, std::multiplies<int>());
  int64_t updateDataNum =
      ((int32_t)outShape.size() == indicesShape.back())
          ? 1
          : (std::accumulate(outShape.begin() + indicesShape.back(), outShape.end(), 1, std::multiplies<int>()));
  int64_t maxIndice = std::accumulate(outShape.begin(), outShape.end() - (outShape.size() - indicesShape.back()), 1,
                                      std::multiplies<int>());
  runParams.maxIndice = maxIndice;
  runParams.indicesLastDim = indicesShape.back();
  int64_t updatesDataEachBlock = BLOCK_SIZE / updatesSize;
  int64_t dataNumOneRepeat = 0;

  for (int64_t i = 0; i + 1 < indicesShape.back(); i++) {
    runParams.varOffSet[i] =
        std::accumulate(outShape.begin() + (i + 1), outShape.end() - (outShape.size() - indicesShape.back()), 1,
                        std::multiplies<int>());
  }

  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : varNum=%ld.", varNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : indicesNum=%ld.", indicesNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdTiling] : updatesNum=%ld.", updatesNum);

  if (updateDataNum < updatesDataEachBlock) {
    runParams.coreNum = 1;
  } else {
    runParams.indiceStep = ceil(float(maxIndice) / coreNum);
    runParams.coreNum = ceil(float(maxIndice) / runParams.indiceStep);
  }

  if (input_dtype == "float32" || input_dtype == "int32") {
    dataNumOneRepeat = 64;
  } else {
    dataNumOneRepeat = 128;
  }

  if (supportAtomic == 1 && input_dtype == "float32") {
    if (CheckScatterNdHighPerfShape(outShape, indicesShape)) {
      CalScatterNdHighPerfBranchParams(runParams, indicesNum, coreNum, ubSize, updateDataNum, updatesDataEachBlock,
                                       indicesSize);
    } else {
      CalAtomicBranchRunningParams(runParams, indicesNum, updatesNum, updateDataNum, ubSize, updatesSize, indicesSize,
                                   updatesDataEachBlock);
    }
  } else {
    CalNotAtomicBranchRunningParams(runParams, varNum, indicesNum, updatesNum, updateDataNum, maxIndice, ubSize,
                                    runParams.coreNum, updatesSize, indicesSize, updatesDataEachBlock,
                                    dataNumOneRepeat);
  }

  SetRuningParams(runParams, runInfo);

  PrintTilingParams(opType, runParams);

  runInfo.block_dim = runParams.coreNum;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  OP_LOGI(opType.c_str(), "ScatterNdTiling run success.");

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(ScatterNd, ScatterNdTiling);
}  // namespace optiling
