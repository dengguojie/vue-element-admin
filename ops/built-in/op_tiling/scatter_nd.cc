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
 * \file scatter_nd.cpp
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

const int64_t UPDATESSIZE_COMPILE_INDEX = 2;
const int64_t INDICESSIZE_COMPILE_INDEX = 3;
const int64_t SUPPORTATOMIC_COMPILE_INDEX = 4;
const int64_t NEEDCAST_COMPILE_INDEX = 5;

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

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num",     "ub_size",        "updates_size",
                                                          "indices_size", "support_atomic", "need_cast"};

void CalAtomicBranchRunningParams(ScatterNdTilingParams& runParams, int64_t indicesNum, int64_t updatesNum,
                                  int64_t updateDataNum, int64_t ubSize, int64_t updatesSize, int64_t indicesSize,
                                  int64_t updatesDataEachBlock) {
  int64_t updateSizeByte = updatesSize * updatesNum;
  int64_t halfUbSize = ubSize / 2;
  OP_TILING_CHECK(halfUbSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "halfUbSize = 0 is not support"),
                  return );
  OP_TILING_CHECK(indicesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "indicesSize = 0 is not support"),
                  return );
  OP_TILING_CHECK(updatesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "updatesSize = 0 is not support"),
                  return );
  OP_TILING_CHECK(runParams.indicesLastDim == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "runParams.indicesLastDim = 0 is not support"),
                  return );
  OP_TILING_CHECK(runParams.indicesLastDim == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "runParams.indicesLastDim = 0 is not support"),
                  return );
  OP_TILING_CHECK(updatesDataEachBlock == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "updatesDataEachBlock = 0 is not support"), return);
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
                                     int64_t updatesNum, int64_t updateDataNum, int64_t ubSize, int64_t coreNum,
                                     int64_t updatesSize, int64_t indicesSize, int64_t updatesDataEachBlock) {
  int64_t varAllSizeByte = updatesSize * varNum;
  int64_t varSizeByte = updatesSize * runParams.indiceStep * updateDataNum;
  int64_t updateSizeByte = updatesSize * updatesNum;
  int64_t varUbSize = ubSize / 8 * 3;
  int64_t indicesUbSize = ubSize / 8 * 2;
  OP_TILING_CHECK(varUbSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "varUbSize = 0 is not support"),
                  return );
  OP_TILING_CHECK(indicesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "indicesSize = 0 is not support"),
                  return );
  OP_TILING_CHECK(updatesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "updatesSize = 0 is not support"),
                  return );
  OP_TILING_CHECK(runParams.indicesLastDim == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "runParams.indicesLastDim = 0 is not support"),
                  return );
  OP_TILING_CHECK(runParams.indicesLastDim == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "runParams.indicesLastDim = 0 is not support"),
                  return );
  OP_TILING_CHECK(updatesDataEachBlock == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "updatesDataEachBlock = 0 is not support"), return);
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
                                      int64_t indicesSize, int64_t updatesSize, int64_t need_cast) {
  const int64_t UB_NUM = 2;
  const int64_t UB_NEEDCAST_NUM = 4;

  int64_t alloc_indice_ubsize = ubSize / UB_NUM;
  if (need_cast == 1) {
    alloc_indice_ubsize = ubSize / UB_NEEDCAST_NUM;
  }
  OP_TILING_CHECK(alloc_indice_ubsize == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "alloc_indice_ubsize = 0 is not support"), return);
  OP_TILING_CHECK(indicesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "indicesSize = 0 is not support"),
                  return);
  OP_TILING_CHECK(coreNum == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "coreNum = 0 is not support"), return);
  OP_TILING_CHECK(updatesDataEachBlock == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "updatesDataEachBlock = 0 is not support"), return );
  OP_TILING_CHECK(updatesSize == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd", "updatesSize = 0 is not support"), return );
  int64_t alloc_ub_indicesnum = alloc_indice_ubsize / indicesSize / runParams.indicesLastDim * runParams.indicesLastDim;
  runParams.tilingMode = TILING_MODE_16;
  runParams.updatesDataNum = updateDataNum;
  runParams.indicesEachCoreData = ceil(float(indicesNum) / coreNum);
  runParams.indicesEachCoreData = (runParams.indicesEachCoreData + runParams.indicesLastDim - 1) /
                                  runParams.indicesLastDim * runParams.indicesLastDim;
  runParams.coreNum = ceil(float(indicesNum) / runParams.indicesEachCoreData);
  runParams.indicesLastCoreData = indicesNum - runParams.indicesEachCoreData * (runParams.coreNum - 1);
  runParams.eachCoreIndicesLoopNum = runParams.indicesEachCoreData / alloc_ub_indicesnum;
  runParams.eachCoreIndicesLastNum = runParams.indicesEachCoreData % alloc_ub_indicesnum;
  runParams.lastCoreIndicesLoopNum = runParams.indicesLastCoreData / alloc_ub_indicesnum;
  runParams.lastCoreIndicesLastNum = runParams.indicesLastCoreData % alloc_ub_indicesnum;
  runParams.updatesLoopNum = updateDataNum / (alloc_indice_ubsize / updatesSize);
  runParams.updatesLastNum = updateDataNum % (alloc_indice_ubsize / updatesSize);
}

void SetRuningParams(const ScatterNdTilingParams& params, utils::OpRunInfo& runInfo) {
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
  runInfo.AddTilingData(params.indicesLastDim);
  for (size_t i = 0; i < params.varOffSet.size(); i++) {
    runInfo.AddTilingData(params.varOffSet[i]);
  }
  runInfo.AddTilingData(params.varEachCoreSetZeroLoopNum);
  runInfo.AddTilingData(params.varEachCoreSetZeroLastNum);
  runInfo.AddTilingData(params.varLastCoreSetZeroLoopNum);
  runInfo.AddTilingData(params.varLastCoreSetZeroLastNum);
  runInfo.AddTilingData(params.indicesEachCoreData);
  runInfo.AddTilingData(params.indicesLastCoreData);
  runInfo.AddTilingData(params.eachCoreIndicesLoopNum);
  runInfo.AddTilingData(params.eachCoreIndicesLastNum);
  runInfo.AddTilingData(params.lastCoreIndicesLoopNum);
  runInfo.AddTilingData(params.lastCoreIndicesLastNum);
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

bool CheckScatterNdTensorShape(const std::string& opType, const GeShape& indicesShape, const GeShape& updatesShape,
                               std::vector<int64_t> outputShape, const int64_t indicesLastDim) {
  const int64_t& indicesDims = indicesShape.GetDimNum();
  const int64_t& updatesDims = updatesShape.GetDimNum();
  const int64_t& outputDims = outputShape.size();

  if (indicesDims <= 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "the ndim of indices is less than 1 or equal to 1");
    return false;
  }

  if (indicesDims - 1 + outputDims - indicesLastDim != updatesDims) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "output's shape and updates'shape are not equal in some dimensions");
    return false;
  }

  for (int64_t i = 0; i < indicesDims - 1; i++) {
    if (indicesShape.GetDim(i) != updatesShape.GetDim(i)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "indices's shape and updates'shape are not equal in some dimensions");
      return false;
    }
  }

  for (int64_t i = 0; i < updatesDims - indicesDims + 1; i++) {
    if (updatesShape.GetDim(indicesDims - 1 + i) != outputShape[indicesLastDim + i]) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "output's shape and updates'shape are not equal in some dimensions");
      return false;
    }
  }
  return true;
}

bool CheckScatterNdHighPerfShape(std::vector<int64_t> outShape, const GeShape& indicesShape) {
  if (indicesShape.GetDimNum() != 2 || outShape.size() != 2 || indicesShape.GetDim(1) != 1) {
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

bool ScatterNdTiling(const std::string& opType, const ge::Operator& opParas, const std::vector<int64_t>& op_info,
                     utils::OpRunInfo& runInfo) {
  OP_LOGI(opType.c_str(), "ScatterNdTiling running.");
  PROFILING_TILING_INIT(opType.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(opParas);

  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_info failed."),
                  return false);

  auto indices_desc = operator_info->MutableInputDesc(0);
  auto x_desc = operator_info->MutableInputDesc(1);
  auto y_desc = operator_info->MutableOutputDesc(0);

  OP_TILING_CHECK(indices_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_indices_info failed."),
                  return false);
  OP_TILING_CHECK(x_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_x_info failed."), return false);
  OP_TILING_CHECK(y_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_y_info failed."), return false);
  OP_TILING_CHECK(COMPILE_INFO_KEY.size() != op_info.size(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "parse op_info failed."), return false);
  const GeShape& indicesShape = indices_desc->MutableShape();
  const GeShape& updatesShape = x_desc->MutableShape();
  const std::vector<int64_t>& outShape = y_desc->MutableShape().GetDims();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  int64_t indicesBack = indicesShape.GetDim(indicesShape.GetDimNum() - 1);
  bool is_valid_shape = CheckScatterNdTensorShape(opType, indicesShape, updatesShape, outShape, indicesBack);
  if (!is_valid_shape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckScatterNdTensorShape is failed");
    return false;
  }
  int64_t coreNum = op_info[0];
  int64_t ubSize = op_info[1];
  int64_t updatesSize = op_info[UPDATESSIZE_COMPILE_INDEX];
  int64_t indicesSize = op_info[INDICESSIZE_COMPILE_INDEX];
  int64_t supportAtomic = op_info[SUPPORTATOMIC_COMPILE_INDEX];
  int64_t need_cast = op_info[NEEDCAST_COMPILE_INDEX];
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  if (coreNum <= ZERO || ubSize <= ZERO || updatesSize <= ZERO || indicesSize <= ZERO) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        opType, "coreNum, ubSize, updatesSize, indicesSize must be greater to 0, but got %ld, %ld, %ld, %ld", coreNum,
        ubSize, updatesSize, indicesSize);
    return false;
  }

  ScatterNdTilingParams runParams;
  InitRunningParams(runParams);

  int64_t varNum = std::accumulate(outShape.begin(), outShape.end(), 1, std::multiplies<int>());
  int64_t indicesNum = GetTensorSize(indicesShape);
  int64_t updatesNum = GetTensorSize(updatesShape);
  int64_t updateDataNum =
      ((int32_t)outShape.size() == indicesBack)
          ? 1
          : (std::accumulate(outShape.begin() + indicesBack, outShape.end(), 1, std::multiplies<int>()));
  int64_t maxIndice =
      std::accumulate(outShape.begin(), outShape.end() - (outShape.size() - indicesBack), 1, std::multiplies<int>());
  runParams.maxIndice = maxIndice;
  runParams.indicesLastDim = indicesBack;
  int64_t updatesDataEachBlock = BLOCK_SIZE / updatesSize;

  for (int64_t i = 0; i < indicesBack; i++) {
    runParams.varOffSet[i] = std::accumulate(
        outShape.begin() + (i + 1), outShape.end() - (outShape.size() - indicesBack), 1, std::multiplies<int>());
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

  if (supportAtomic == 1) {
    CalScatterNdHighPerfBranchParams(runParams, indicesNum, coreNum, ubSize, updateDataNum, updatesDataEachBlock,
                                     indicesSize, updatesSize, need_cast);
  } else {
    CalNotAtomicBranchRunningParams(runParams, varNum, indicesNum, updatesNum, updateDataNum, ubSize, runParams.coreNum,
                                    updatesSize, indicesSize, updatesDataEachBlock);
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  SetRuningParams(runParams, runInfo);
  PrintTilingParams(opType, runParams);
  runInfo.SetBlockDim(runParams.coreNum);
  OP_LOGI(opType.c_str(), "ScatterNdTiling run success.");
  PROFILING_TILING_END();
  return true;
}

REGISTER_OP_TILING_V3_WITH_VECTOR(ScatterNd, ScatterNdTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
