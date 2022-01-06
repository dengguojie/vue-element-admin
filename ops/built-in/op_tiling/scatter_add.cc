/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \brief
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"

namespace optiling {
const int64_t BLOCK_SIZE = 32;
const int64_t OUT_SPECIAL_DIM_0 = 163623;
const int64_t OUT_SPECIAL_DIM_1 = 1;
const int64_t OUT_SPECIAL_DIM_2 = 80;
const int64_t OUT_SPECIAL_DIM_3 = 21340;
const int64_t OUT_SPECIAL_DIM_4 = 12828;
const int64_t CORENUM_IDX = 0;
const int64_t UBSIZE_IDX = 1;
const int64_t VARSIZE_IDX = 2;
const int64_t INDICESSIZE_IDX = 3;
const int64_t AUTOMIC_IDX = 4;
const int64_t INPUTDESC_IDX = 2;
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

struct ScatterAddTilingParams {
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
  int64_t indicesEachCoreData;
  int64_t indicesLastCoreData;
  int64_t eachCoreIndicesLoopNum;
  int64_t eachCoreIndicesLastNum;
  int64_t lastCoreIndicesLoopNum;
  int64_t lastCoreIndicesLastNum;
};

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "ub_size", "var_size", "indices_size",
                                                          "support_atomic"};

void InitRunningParams(ScatterAddTilingParams& params) {
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
  params.indicesEachCoreData = 0;
  params.indicesLastCoreData = 0;
  params.eachCoreIndicesLoopNum = 0;
  params.eachCoreIndicesLastNum = 0;
  params.lastCoreIndicesLoopNum = 0;
  params.lastCoreIndicesLastNum = 0;
}

void CalAtomicBranchRunningParams(ScatterAddTilingParams& runParams, int64_t indicesNum, int64_t updatesNum,
                                  int64_t updateDataNum, int64_t ubSize, int64_t varSize, int64_t indicesSize,
                                  int64_t varDataEachBlock) {
  int64_t updateSizeByte = varSize * updatesNum;
  int64_t halfUbSize = ubSize / 2;
  OP_TILING_CHECK(halfUbSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "halfUbSize = 0 is not support"),
                  return);
  OP_TILING_CHECK(varSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "varSize = 0 is not support"), return);
  OP_TILING_CHECK(indicesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "indicesSize = 0 is not support"),
                  return);
  runParams.updatesLoopNum = updateDataNum / (halfUbSize / varSize);
  runParams.updatesLastNum = updateDataNum % (halfUbSize / varSize);
  runParams.indicesLoopNum = indicesNum / (halfUbSize / indicesSize);
  runParams.indicesLastNum = indicesNum % (halfUbSize / indicesSize);
  runParams.updatesDataNum = updateDataNum;
  runParams.updatesNum = updatesNum;

  if (updateDataNum % varDataEachBlock == 0) {
    if (updateSizeByte <= halfUbSize) {
      runParams.tilingMode = TILING_MODE_1;
    } else {
      runParams.tilingMode = TILING_MODE_2;
    }
  } else {
    if (updateDataNum < varDataEachBlock) {
      if (updateSizeByte <= halfUbSize) {
        runParams.tilingMode = TILING_MODE_4;
        runParams.updatesLoopNum = updatesNum / (halfUbSize / varSize);
        runParams.updatesLastNum = updatesNum % (halfUbSize / varSize);
      } else {
        runParams.tilingMode = TILING_MODE_4;
      }
    } else {
      runParams.tilingMode = TILING_MODE_5;
    }
  }
}

void CalNotAtomicBranchRunningParams(ScatterAddTilingParams& runParams, int64_t varNum, int64_t indicesNum,
                                     int64_t updatesNum, int64_t updateDataNum, int64_t ubSize, int64_t coreNum,
                                     int64_t varSize, int64_t indicesSize, int64_t varDataEachBlock) {
  int64_t varAllSizeByte = varSize * varNum;
  int64_t varSizeByte = varSize * runParams.indiceStep * updateDataNum;
  int64_t updateSizeByte = varSize * updatesNum;
  int64_t varUbSize = ubSize / 8 * 3;
  int64_t indicesUbSize = ubSize / 8 * 2;
  OP_TILING_CHECK(varSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "varSize = 0 is not support"), return);
  OP_TILING_CHECK(indicesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "indicesSize = 0 is not support"),
                  return);
  OP_TILING_CHECK(varUbSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "varUbSize = 0 is not support"),
                  return);
  OP_TILING_CHECK(indicesUbSize == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "indicesUbSize = 0 is not support"), return);
  OP_TILING_CHECK(varDataEachBlock == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "varDataEachBlock = 0 is not support"), return);
  runParams.varLoopNum = varNum / (varUbSize / varSize);
  runParams.varLastNum = varNum % (varUbSize / varSize);
  runParams.updatesLoopNum = updateDataNum / (varUbSize / varSize);
  runParams.updatesLastNum = updateDataNum % (varUbSize / varSize);
  runParams.indicesLoopNum = indicesNum / (indicesUbSize / indicesSize);
  runParams.indicesLastNum = indicesNum % (indicesUbSize / indicesSize);
  runParams.updatesDataNum = updateDataNum;
  runParams.updatesNum = updatesNum;
  runParams.varNum = varNum;

  if (updateDataNum % varDataEachBlock == 0) {
    if (updateSizeByte <= varUbSize && varSizeByte <= varUbSize) {
      runParams.tilingMode = TILING_MODE_6;
    } else if (updateSizeByte > varUbSize && varSizeByte <= varUbSize) {
      runParams.tilingMode = TILING_MODE_7;
    } else if (updateSizeByte <= varUbSize && varSizeByte > varUbSize) {
      runParams.tilingMode = TILING_MODE_8;
    } else {
      runParams.tilingMode = TILING_MODE_9;
    }
  } else if (updateDataNum < varDataEachBlock) {
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
    if (updateDataNum / (varUbSize / varSize) == 0) {
      runParams.tilingMode = TILING_MODE_14;
    } else {
      runParams.tilingMode = TILING_MODE_15;
    }
  }

  if (runParams.tilingMode == TILING_MODE_6 || runParams.tilingMode == TILING_MODE_7) {
    runParams.varEachCoreData = runParams.indiceStep * runParams.updatesDataNum;
    int64_t varLastCoreData = varNum - runParams.varEachCoreData * (coreNum - 1);
    runParams.varEachCoreBurstLen = runParams.varEachCoreData / varDataEachBlock;
    runParams.varLastCoreBurstLen = varLastCoreData / varDataEachBlock;
  }
  if (runParams.tilingMode == TILING_MODE_9 || runParams.tilingMode == TILING_MODE_14 ||
      runParams.tilingMode == TILING_MODE_15) {
    runParams.varLoopNum = updateDataNum / (varUbSize / varSize);
    runParams.varLastNum = updateDataNum % (varUbSize / varSize);
  }
}

void CalScatterAddHighPerfBranchParams(ScatterAddTilingParams& runParams, int64_t indicesNum, int64_t coreNum,
                                       int64_t ubSize, int64_t updateDataNum, int64_t varDataEachBlock,
                                       int64_t indicesSize) {
  int64_t halfUbSize = ubSize / 2;
  OP_TILING_CHECK(halfUbSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "halfUbSize = 0 is not support"),
                  return);
  OP_TILING_CHECK(coreNum == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "coreNum = 0 is not support"), return);
  OP_TILING_CHECK(indicesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "indicesSize = 0 is not support"),
                  return);
  OP_TILING_CHECK(varDataEachBlock == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_add", "varDataEachBlock = 0 is not support"), return);
  runParams.tilingMode = TILING_MODE_16;
  runParams.updatesDataNum = updateDataNum;
  runParams.indicesEachCoreData = ceil(float(indicesNum) / coreNum);
  runParams.coreNum = ceil(float(indicesNum) / runParams.indicesEachCoreData);
  runParams.indicesLastCoreData = indicesNum - runParams.indicesEachCoreData * (runParams.coreNum - 1);
  runParams.eachCoreIndicesLoopNum = runParams.indicesEachCoreData / (halfUbSize / indicesSize);
  runParams.eachCoreIndicesLastNum = runParams.indicesEachCoreData % (halfUbSize / indicesSize);
  runParams.lastCoreIndicesLoopNum = runParams.indicesLastCoreData / (halfUbSize / indicesSize);
  runParams.lastCoreIndicesLastNum = runParams.indicesLastCoreData % (halfUbSize / indicesSize);

  if (updateDataNum % varDataEachBlock == 0) {
    runParams.tilingMode = TILING_MODE_17;
  }
}

void SetRuningParams(const ScatterAddTilingParams& params, utils::OpRunInfo& runInfo) {
  runInfo.AddTilingData(params.tilingMode);
  runInfo.AddTilingData(params.indiceStep);
  runInfo.AddTilingData(params.coreNum);
  runInfo.AddTilingData(params.updatesDataNum);
  runInfo.AddTilingData(params.indicesLoopNum);
  runInfo.AddTilingData(params.indicesLastNum);
  runInfo.AddTilingData(params.updatesNum);
  runInfo.AddTilingData(params.updatesLoopNum);
  runInfo.AddTilingData(params.updatesLastNum);
  runInfo.AddTilingData(params.varNum);
  runInfo.AddTilingData(params.varLoopNum);
  runInfo.AddTilingData(params.varLastNum);
  runInfo.AddTilingData(params.varEachCoreBurstLen);
  runInfo.AddTilingData(params.varLastCoreBurstLen);
  runInfo.AddTilingData(params.maxIndice);
  runInfo.AddTilingData(params.varEachCoreData);
  runInfo.AddTilingData(params.indicesEachCoreData);
  runInfo.AddTilingData(params.indicesLastCoreData);
  runInfo.AddTilingData(params.eachCoreIndicesLoopNum);
  runInfo.AddTilingData(params.eachCoreIndicesLastNum);
  runInfo.AddTilingData(params.lastCoreIndicesLoopNum);
  runInfo.AddTilingData(params.lastCoreIndicesLastNum);
}

void PrintTilingParams(const std::string& opType, const ScatterAddTilingParams& params) {
  OP_LOGD(opType, "op [ScatterAddTiling] : tilingMode=%ld. ", params.tilingMode);
  OP_LOGD(opType, "op [ScatterAddTiling] : indiceStep=%ld. ", params.indiceStep);
  OP_LOGD(opType, "op [ScatterAddTiling] : coreNum=%ld. ", params.coreNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : updatesDataNum=%ld.", params.updatesDataNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : indicesLoopNum=%ld.", params.indicesLoopNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : indicesLastNum=%ld.", params.indicesLastNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : updatesNum=%ld.", params.updatesNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : updatesLoopNum=%ld.", params.updatesLoopNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : updatesLastNum=%ld.", params.updatesLastNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : varNum=%ld.", params.varNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : varLoopNum=%ld.", params.varLoopNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : varLastNum=%ld.", params.varLastNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : varEachCoreBurstLen=%ld.", params.varEachCoreBurstLen);
  OP_LOGD(opType, "op [ScatterAddTiling] : varLastCoreBurstLen=%ld.", params.varLastCoreBurstLen);
  OP_LOGD(opType, "op [ScatterAddTiling] : maxIndice=%ld.", params.maxIndice);
  OP_LOGD(opType, "op [ScatterAddTiling] : varEachCoreData=%ld.", params.varEachCoreData);
  OP_LOGD(opType, "op [ScatterAddTiling] : indicesEachCoreData=%ld.", params.indicesEachCoreData);
  OP_LOGD(opType, "op [ScatterAddTiling] : indicesLastCoreData=%ld.", params.indicesLastCoreData);
  OP_LOGD(opType, "op [ScatterAddTiling] : eachCoreIndicesLoopNum=%ld.", params.eachCoreIndicesLoopNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : eachCoreIndicesLastNum=%ld.", params.eachCoreIndicesLastNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : lastCoreIndicesLoopNum=%ld.", params.lastCoreIndicesLoopNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : lastCoreIndicesLastNum=%ld.", params.lastCoreIndicesLastNum);
}

bool CheckScatterAddShape(const std::string& opType, const GeShape& varShape, const GeShape& indicesShape,
                          const GeShape& updatesShape, const GeShape& outShape) {
  if (!(varShape == outShape)) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "the length of var must be same as the length of output.");
    return false;
  }

  if (indicesShape.GetDimNum() == 1 && indicesShape.GetDim(0) == 1 &&
      varShape.GetDimNum() - updatesShape.GetDimNum() == 1) {
    OP_LOGI(opType.c_str(), "Input indices is a scalar.");
    return true;
  }

  int64_t varSize = varShape.GetDimNum();
  int64_t indicesSize = indicesShape.GetDimNum();
  int64_t updatesSize = updatesShape.GetDimNum();
  if (varSize + indicesSize - 1 != updatesSize) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        opType, "varSize and indicesSize does not satisfy the relation expression with updatesSize.");
    return false;
  }
  for (int64_t i = 0; i < indicesSize; i++) {
    if (indicesShape.GetDim(i) != updatesShape.GetDim(i)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "indicesShapeDim does not equal to updatesShapeDim.");
      return false;
    }
  }
  for (int64_t i = indicesSize + 1; i < updatesSize; i++) {
    if (varShape.GetDim(i - indicesSize + 1) != updatesShape.GetDim(i)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "varShapeGetDim does not equal to updatesShapeDim.");
      return false;
    }
  }
  return true;
}

bool CheckScatterAddHighPerfShape(const GeShape& varShape, const GeShape& indicesShape) {
  if ((indicesShape.GetDimNum() == 1 && varShape.GetDimNum() == 2) &&
      ((varShape.GetDim(0) == OUT_SPECIAL_DIM_0 && varShape.GetDim(1) == OUT_SPECIAL_DIM_1) ||
       (varShape.GetDim(0) == OUT_SPECIAL_DIM_0 && varShape.GetDim(1) == OUT_SPECIAL_DIM_2) ||
       (varShape.GetDim(0) == OUT_SPECIAL_DIM_3 && varShape.GetDim(1) == OUT_SPECIAL_DIM_1) ||
       (varShape.GetDim(0) == OUT_SPECIAL_DIM_3 && varShape.GetDim(1) == OUT_SPECIAL_DIM_2))) {
    return true;
  }
  if ((indicesShape.GetDimNum() == 1 && varShape.GetDimNum() == 1) && (varShape.GetDim(0) == OUT_SPECIAL_DIM_4)) {
    return true;
  }
  return false;
}

bool GetScatterAddCompileParams(const std::string& opType, const std::vector<int64_t>& opCompileInfo, int64_t& coreNum,
                                int64_t& ubSize, int64_t& varSize, int64_t& indicesSize, int64_t& supportAtomic) {
  OP_TILING_CHECK(COMPILE_INFO_KEY.size() != opCompileInfo.size(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "parse opCompileInfo failed."), return false);
  coreNum = opCompileInfo[CORENUM_IDX];
  ubSize = opCompileInfo[UBSIZE_IDX];
  varSize = opCompileInfo[VARSIZE_IDX];
  indicesSize = opCompileInfo[INDICESSIZE_IDX];
  supportAtomic = opCompileInfo[AUTOMIC_IDX];
  return true;
}

bool ScatterAddTiling(const std::string& opType, const ge::Operator& opParas, const std::vector<int64_t>& opCompileInfo,
                      utils::OpRunInfo& runInfo) {
  using namespace ge;

  OP_LOGI(opType.c_str(), "ScatterAddTiling running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(opParas);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_info failed."),
                  return false);
  // get input var Desc
  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  const GeShape& varShape = input_desc->MutableShape();
  const ge::DataType input_dtype = input_desc->GetDataType();
  // get input indices Desc
  input_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  const GeShape& indicesShape = input_desc->MutableShape();
  // get input updates Desc
  input_desc = operator_info->MutableInputDesc(INPUTDESC_IDX);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  const GeShape& updatesShape = input_desc->MutableShape();
  // get output Desc
  auto output_desc = operator_info->MutableOutputDesc(0);
  OP_TILING_CHECK(output_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get output_desc failed."),
                  return false);
  const GeShape& outShape = output_desc->MutableShape();

  bool is_valid_shape = CheckScatterAddShape(opType, varShape, indicesShape, updatesShape, outShape);
  if (!is_valid_shape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckScatterAddShape is failed.");
    return false;
  }

  int64_t coreNum = 0;
  int64_t ubSize = 0;
  int64_t varSize = 0;
  int64_t indicesSize = 0;
  int64_t supportAtomic = 0;

  OP_TILING_CHECK(
      !GetScatterAddCompileParams(opType, opCompileInfo, coreNum, ubSize, varSize, indicesSize, supportAtomic),
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileInfo errror."), return false);

  if (coreNum <= ZERO || ubSize <= ZERO || varSize <= ZERO || indicesSize <= ZERO) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        opType, "coreNum, ubSize, varSize, indicesSize must be greater to 0, but got %ld, %ld, %ld, %ld", coreNum,
        ubSize, varSize, indicesSize);
    return false;
  }
  ScatterAddTilingParams runParams;
  InitRunningParams(runParams);
  int64_t varNum = varShape.GetShapeSize();
  int64_t indicesNum = indicesShape.GetShapeSize();
  int64_t updatesNum = updatesShape.GetShapeSize();
  int64_t updateDataNum = 1;
  int64_t varDimNum = varShape.GetDimNum();
  if (varDimNum > 1) {
    for (int64_t i = 1; i < varDimNum; i++) {
      updateDataNum *= varShape.GetDim(i);
    }
  }
  int64_t maxIndice = varShape.GetDim(0);
  runParams.maxIndice = maxIndice;
  int64_t varDataEachBlock = BLOCK_SIZE / varSize;

  OP_LOGD(opType, "op [ScatterAddTiling] : varNum=%ld.", varNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : indicesNum=%ld.", indicesNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : updatesNum=%ld.", updatesNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : updateDataNum=%ld.", updateDataNum);
  OP_LOGD(opType, "op [ScatterAddTiling] : maxIndice=%ld.", maxIndice);

  if (updateDataNum < varDataEachBlock) {
    runParams.coreNum = 1;
  } else {
    runParams.indiceStep = ceil(float(maxIndice) / coreNum);
    runParams.coreNum = ceil(float(maxIndice) / runParams.indiceStep);
  }

  if (supportAtomic == 1 && input_dtype == ge::DT_FLOAT) {
    if (CheckScatterAddHighPerfShape(varShape, indicesShape)) {
      CalScatterAddHighPerfBranchParams(runParams, indicesNum, coreNum, ubSize, updateDataNum, varDataEachBlock,
                                        indicesSize);
    } else {
      runParams.indiceStep = ceil(float(maxIndice) / coreNum);
      runParams.coreNum = ceil(float(maxIndice) / runParams.indiceStep);
      CalAtomicBranchRunningParams(runParams, indicesNum, updatesNum, updateDataNum, ubSize, varSize, indicesSize,
                                   varDataEachBlock);
    }
  } else {
    CalNotAtomicBranchRunningParams(runParams, varNum, indicesNum, updatesNum, updateDataNum, ubSize, runParams.coreNum,
                                    varSize, indicesSize, varDataEachBlock);
  }
  SetRuningParams(runParams, runInfo);
  PrintTilingParams(opType, runParams);

  runInfo.SetBlockDim(runParams.coreNum);

  OP_LOGI(opType.c_str(), "ScatterAddTiling run success.");
  return true;
}

REGISTER_OP_TILING_V3_WITH_VECTOR(ScatterAdd, ScatterAddTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
