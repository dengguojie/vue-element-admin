/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file gather_nd.cpp
 * \brief tiling function of op
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"

namespace optiling {
const int32_t BLOCK_SIZE = 32;
const int32_t PARAMS_SUFFIX_INDEX = 19;
const int32_t PARAMS_CACHED_UB = 100 * 1024;
const int32_t RESERVED_UB_SIZE = 6 * 1024;
const int32_t LAST_DIM_MAX = 8;
const int32_t UB_2K_SIZE = 2 * 1024;

// 1. one params row size is smaller than 32B
// params is not cache in UB
const int32_t TILING_MODE_1 = 1;
// params is cache in UB
const int32_t TILING_MODE_2 = 2;

// 2. one params row size is greater than or equal to 32B
// paramsRow is 32B aligned, params is cache in L1
const int32_t TILING_MODE_3 = 3;
// paramsRow is 32B aligned, params is not cache in UB or L1
const int32_t TILING_MODE_4 = 4;
// paramsRow is 32B aligned, params is cache in UB
const int32_t TILING_MODE_5 = 5;
// paramsRow is not 32B aligned
const int32_t TILING_MODE_6 = 6;
// the data of one params row can not store in half UB, need tiling
const int32_t TILING_MODE_7 = 7;
// special indices shape: complete params data needs to be moved for one indice
const int32_t TILING_MODE_8 = 8;
// the data of one params row can not store in half UB, need tiling and special indices shape: [1]
const int32_t TILING_MODE_9 = 9;
const int32_t INDEX_L1SIZE = 2;
const int32_t INDEX_PARAMSDSIZE = 3;
const int32_t INDEX_INDICESDSIZE = 4;

struct GatherNdParam {
  int32_t tilingMode;

  int32_t needCoreNum;
  int32_t tailProcessCore;
  int32_t indicesNumEachCore;
  int32_t indicesNumRemaining;
  int32_t indicesLoopNum;
  int32_t indicesRowNumOnce;
  int32_t indicesRowNumLast;
  int32_t rowNumOnceUb;
  int32_t rowNumOnceTailUb;
  int32_t innerLoopNum;
  int32_t rowNumLastTailUb;
  int32_t innerLoopNumLast;

  int32_t paramsRow;
  int32_t indicesLastDim;
  int32_t paramsTotal;
  int32_t oneRowLoop;
  int32_t oneRowTail;
};

bool CheckTensorShape(const std::string& opType, GeShape& indicesShape, int32_t indicesLastDim) {
  int32_t indicesDims = indicesShape.GetDimNum();
  if (indicesLastDim == 0 && indicesDims == 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckTensorShape, indices.shape is invalid.");
    return false;
  }

  for (int32_t i = 0; i < indicesDims - 1; i++) {
    if (indicesShape.GetDim(i) <= 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckTensorShape, indices.shape[i] must be > 0");
      return false;
    }
  }
  return true;
}

int GetMaxApproximate(int x, int y) {
  int z = y;
  while (x % y != 0) {
    z = x % y;
    x = y;
    y = z;
  }
  return z;
}

void InitRunningParams(GatherNdParam& params) {
  params.tilingMode = TILING_MODE_1;
  params.needCoreNum = 0;
  params.tailProcessCore = 0;
  params.indicesNumEachCore = 0;
  params.indicesNumRemaining = 0;
  params.indicesLoopNum = 0;
  params.indicesRowNumOnce = 0;
  params.indicesRowNumLast = 0;
  params.rowNumOnceUb = 0;
  params.rowNumOnceTailUb = 0;
  params.innerLoopNum = 0;
  params.rowNumLastTailUb = 0;
  params.innerLoopNumLast = 0;

  params.paramsRow = 0;
  params.indicesLastDim = 0;
  params.paramsTotal = 0;
  params.oneRowLoop = 0;
  params.oneRowTail = 0;
}

void SetRunningParams(const GatherNdParam& runParams, utils::OpRunInfo& runInfo) {
  runInfo.AddTilingData(runParams.tilingMode);
  runInfo.AddTilingData(runParams.needCoreNum);
  runInfo.AddTilingData(runParams.tailProcessCore);
  runInfo.AddTilingData(runParams.indicesNumEachCore);
  runInfo.AddTilingData(runParams.indicesNumRemaining);
  runInfo.AddTilingData(runParams.indicesLoopNum);
  runInfo.AddTilingData(runParams.indicesRowNumOnce);
  runInfo.AddTilingData(runParams.indicesRowNumLast);
  runInfo.AddTilingData(runParams.rowNumOnceUb);
  runInfo.AddTilingData(runParams.rowNumOnceTailUb);
  runInfo.AddTilingData(runParams.innerLoopNum);
  runInfo.AddTilingData(runParams.rowNumLastTailUb);
  runInfo.AddTilingData(runParams.innerLoopNumLast);
  runInfo.AddTilingData(runParams.paramsRow);
  runInfo.AddTilingData(runParams.indicesLastDim);
  runInfo.AddTilingData(runParams.paramsTotal);
  runInfo.AddTilingData(runParams.oneRowLoop);
  runInfo.AddTilingData(runParams.oneRowTail);
}

void PrintGatherNdrunParams(const GatherNdParam& runParams) {
  OP_LOGD("GatherNd", "op [GatherNdTiling] : tilingMode=%d.", runParams.tilingMode);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : needCoreNum=%d.", runParams.needCoreNum);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : tailProcessCore=%d.", runParams.tailProcessCore);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : indicesNumEachCore=%d.", runParams.indicesNumEachCore);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : indicesNumRemaining=%d.", runParams.indicesNumRemaining);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : indicesLoopNum=%d.", runParams.indicesLoopNum);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : indicesRowNumOnce=%d.", runParams.indicesRowNumOnce);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : indicesRowNumLast=%d.", runParams.indicesRowNumLast);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : rowNumOnceUb=%d.", runParams.rowNumOnceUb);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : rowNumOnceTailUb=%d.", runParams.rowNumOnceTailUb);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : innerLoopNum=%d.", runParams.innerLoopNum);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : rowNumLastTailUb=%d.", runParams.rowNumLastTailUb);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : innerLoopNumLast=%d.", runParams.innerLoopNumLast);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : paramsRow=%d.", runParams.paramsRow);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : indicesLastDim=%d.", runParams.indicesLastDim);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : paramsTotal=%d.", runParams.paramsTotal);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : oneRowLoop=%d.", runParams.oneRowLoop);
  OP_LOGD("GatherNd", "op [GatherNdTiling] : oneRowTail=%d.", runParams.oneRowTail);
}

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "ub_size", "l1_size", "params_dsize",
                                                          "indices_dsize"};

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool GatherNdTiling(const std::string& opType, const ge::Operator& opParas, const std::vector<int64_t>& op_info,
                    utils::OpRunInfo& runInfo) {
  PROFILING_TILING_INIT(opType.c_str());
  OP_LOGD(opType.c_str(), "op[GatherNdTiling] tiling running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(opParas);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  GeShape& paramsShape = input_desc->MutableShape();

  input_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  GeShape& indicesShape = input_desc->MutableShape();

  auto output_desc = operator_info->MutableOutputDesc(0);
  OP_TILING_CHECK(output_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get output_desc failed."),
                  return false);
  std::vector<int64_t> yShape = output_desc->MutableShape().GetDims();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  int32_t paramsDims = paramsShape.GetDimNum();
  int32_t indicesDims = indicesShape.GetDimNum();
  if (indicesDims < 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op GatherNdTiling: indices dim is invalid.");
    return false;
  }

  int32_t indicesLastDim = indicesShape.GetDim(indicesDims - 1);
  // indices.shape[-1] must be <= params.rank, and shape only support 1D ~ 8D
  if (indicesLastDim > paramsDims || indicesLastDim > LAST_DIM_MAX || indicesLastDim < 0) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices",
                                  "the last dim of indices is more than the dim of x, "
                                  "or the last dim of indices is greater than 8 or less than 0");
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op GatherNdTiling: the last dim of indices shape is invalid.");
    return false;
  }

  // check inputs shape
  bool ret = CheckTensorShape(opType, indicesShape, indicesLastDim);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op GatherNdTiling: inputs shape are invalid.");
    return ret;
  }

  // get compile info
  OP_TILING_CHECK(COMPILE_INFO_KEY.size() != op_info.size(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "parse op_info failed."), return false);
  int32_t coreNum = static_cast<int32_t>(op_info[0]);
  int32_t ubSize = static_cast<int32_t>(op_info[1]);
  int32_t l1Size = static_cast<int32_t>(op_info[INDEX_L1SIZE]);
  int32_t paramsDSize = static_cast<int32_t>(op_info[INDEX_PARAMSDSIZE]);
  int32_t indicesDSize = static_cast<int32_t>(op_info[INDEX_INDICESDSIZE]);
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  int32_t availableUbSize = ubSize - RESERVED_UB_SIZE;
  int32_t halfUbSize = availableUbSize / 2;

  int32_t paramsTotal = paramsShape.GetShapeSize();
  int32_t indicesTotal = indicesShape.GetShapeSize();
  int32_t paramsTotalTmp = paramsTotal;
  // e.g. paramsShape:(2, 100, 4, 3) => paramsSuffixList:[100*4*3, 4*3, 3, 1]
  std::vector<int32_t> paramsSuffixList;
  for (int i = 0; i < indicesLastDim; i++) {
    paramsTotalTmp = paramsTotalTmp / paramsShape.GetDim(i);
    paramsSuffixList.push_back(paramsTotalTmp);
  }

  // init running parameters
  GatherNdParam runParams;
  InitRunningParams(runParams);
  runParams.indicesLastDim = indicesLastDim;
  runParams.paramsTotal = paramsTotal;

  runParams.paramsRow = 1;
  for (int i = indicesLastDim; i < paramsDims; i++) {
    runParams.paramsRow *= paramsShape.GetDim(i);
  }
  OP_LOGD("GatherNd", "op [GatherNdTiling] : paramsDims=%d, indicesDims=%d, indicesLastDim=%d, paramsRow=%d",
          paramsDims, indicesDims, indicesLastDim, runParams.paramsRow);
  int64_t indicesPrefix = indicesTotal / indicesLastDim;

  // block tiling: indices tiling
  runParams.needCoreNum = coreNum;
  runParams.indicesNumEachCore = (indicesPrefix) / runParams.needCoreNum;
  runParams.indicesNumRemaining = (indicesPrefix) % runParams.needCoreNum;
  if (indicesPrefix <= runParams.needCoreNum) {
    runParams.needCoreNum = indicesPrefix;
    runParams.tailProcessCore = 0;
    runParams.indicesNumEachCore = 1;
    runParams.indicesNumRemaining = 0;
  }

  int32_t blockNum = BLOCK_SIZE / paramsDSize;
  int32_t paramsTotalCeil = (paramsTotal + blockNum - 1) / blockNum * blockNum;
  int32_t tilingMode = 0;
  // complete params data needs to be moved for one indice
  if (indicesLastDim == 0) {
    tilingMode = TILING_MODE_8;

    if (paramsTotal < blockNum) {
      runParams.needCoreNum = 1;
      runParams.tailProcessCore = 0;
      runParams.indicesNumEachCore = indicesPrefix;
      runParams.indicesNumRemaining = 0;
    }

    int32_t ubParamsElem = (ubSize - UB_2K_SIZE) / paramsDSize;
    runParams.innerLoopNum = paramsTotal / ubParamsElem;
    runParams.rowNumLastTailUb = paramsTotal % ubParamsElem;
    if (runParams.innerLoopNum >= 1 && runParams.rowNumLastTailUb <= blockNum) {
      runParams.innerLoopNum = runParams.innerLoopNum - 1;
      runParams.rowNumLastTailUb = ubParamsElem + runParams.rowNumLastTailUb;
    }
  } else {
    int32_t halfUbIndicesElem = halfUbSize / indicesDSize;
    int32_t halfUbIndicesNum = halfUbIndicesElem / indicesLastDim;
    int32_t halfUbParamsElem = halfUbSize / paramsDSize;
    int32_t halfRemainUbSize = (availableUbSize - PARAMS_CACHED_UB) / 2;
    int32_t halfRemainIndicesElem = halfRemainUbSize / indicesDSize;
    int32_t halfRemainParamsElem = halfRemainUbSize / paramsDSize;

    int32_t indicesNumPerLoop = halfUbIndicesNum;
    int32_t paramsElemPerUb = halfUbParamsElem;
    int32_t proCoreNum = GetMaxApproximate(indicesShape.GetDim(0), coreNum);

    if (runParams.paramsRow < blockNum) {
      if (runParams.indicesNumEachCore * runParams.paramsRow <= blockNum) {
        if (indicesShape.GetDim(0) <= coreNum &&
            indicesPrefix / indicesShape.GetDim(0) * runParams.paramsRow >= blockNum) {
          runParams.needCoreNum = indicesShape.GetDim(0);
          runParams.tailProcessCore = 0;
          runParams.indicesNumEachCore = indicesPrefix / runParams.needCoreNum;
          runParams.indicesNumRemaining = 0;
        } else {
          if (indicesShape.GetDim(0) > coreNum && indicesPrefix / proCoreNum * runParams.paramsRow >= blockNum) {
            runParams.needCoreNum = proCoreNum;
            runParams.tailProcessCore = 0;
            runParams.indicesNumEachCore = indicesPrefix / runParams.needCoreNum;
            runParams.indicesNumRemaining = 0;
          } else {
            runParams.needCoreNum = 1;
            runParams.tailProcessCore = 0;
            runParams.indicesNumEachCore = indicesPrefix;
            runParams.indicesNumRemaining = 0;
          }
        }
      }

      if (paramsTotalCeil <= PARAMS_CACHED_UB / paramsDSize) {
        tilingMode = TILING_MODE_2;
      } else {
        tilingMode = TILING_MODE_1;
      }

      if (tilingMode == TILING_MODE_2) {
        indicesNumPerLoop = halfRemainIndicesElem / indicesLastDim;
        paramsElemPerUb = halfRemainParamsElem;
        OP_LOGD("GatherNd", "op [GatherNdTiling] : indicesNumPerLoop=%d, paramsElemPerUb=%d.", indicesNumPerLoop,
                paramsElemPerUb);
      }

      runParams.indicesLoopNum = runParams.indicesNumEachCore / indicesNumPerLoop;
      runParams.indicesRowNumOnce = indicesNumPerLoop;
      if (runParams.indicesNumEachCore % runParams.indicesRowNumOnce != 0) {
        runParams.indicesRowNumLast = runParams.indicesNumEachCore % runParams.indicesRowNumOnce;
      }

      runParams.rowNumOnceUb = paramsElemPerUb / runParams.paramsRow;
      if (runParams.rowNumOnceUb % blockNum != 0) {
        runParams.rowNumOnceUb = runParams.rowNumOnceUb - runParams.rowNumOnceUb % blockNum;
      }

      runParams.innerLoopNum = runParams.indicesRowNumOnce / runParams.rowNumOnceUb;
      if (runParams.indicesRowNumOnce % runParams.rowNumOnceUb != 0) {
        runParams.rowNumOnceTailUb = runParams.indicesRowNumOnce % runParams.rowNumOnceUb;
      }
      if (runParams.innerLoopNum > 0 && runParams.rowNumOnceTailUb > 0 &&
          runParams.rowNumOnceTailUb * runParams.paramsRow < blockNum) {
        runParams.innerLoopNum = runParams.innerLoopNum - 1;
        runParams.rowNumOnceTailUb = runParams.rowNumOnceTailUb + runParams.rowNumOnceUb;
      }

      runParams.innerLoopNumLast = runParams.indicesRowNumLast / runParams.rowNumOnceUb;
      if (runParams.indicesRowNumLast % runParams.rowNumOnceUb != 0) {
        runParams.rowNumLastTailUb = runParams.indicesRowNumLast % runParams.rowNumOnceUb;
      }
      if (runParams.innerLoopNumLast > 0 && runParams.rowNumLastTailUb > 0 &&
          runParams.rowNumLastTailUb * runParams.paramsRow < blockNum) {
        runParams.innerLoopNumLast = runParams.innerLoopNumLast - 1;
        runParams.rowNumLastTailUb = runParams.rowNumLastTailUb + runParams.rowNumOnceUb;
      }
    } else {  // paramsRow size >= 32
      int32_t paramsRowCeil = (runParams.paramsRow + blockNum - 1) / blockNum * blockNum;
      if (paramsRowCeil <= halfUbParamsElem) {
        if (runParams.paramsRow % blockNum == 0) {  // 32B aligned
          if (paramsTotalCeil <= PARAMS_CACHED_UB / paramsDSize && paramsRowCeil <= halfRemainParamsElem) {
            tilingMode = TILING_MODE_5;  // params cached in ub
          } else if (paramsTotalCeil <= l1Size / paramsDSize) {
            tilingMode = TILING_MODE_3;  // params cached in L1
          } else {
            tilingMode = TILING_MODE_4;  // params in gm
          }
        } else {  // not 32B aligned
          tilingMode = TILING_MODE_6;
        }

        if (tilingMode == TILING_MODE_5) {
          indicesNumPerLoop = halfRemainIndicesElem / indicesLastDim;
          paramsElemPerUb = halfRemainParamsElem;
          OP_LOGD("GatherNd", "op [GatherNdTiling] : indicesNumPerLoop=%d, paramsElemPerUb=%d.", indicesNumPerLoop,
                  paramsElemPerUb);
        }

        runParams.indicesLoopNum = runParams.indicesNumEachCore / indicesNumPerLoop;
        runParams.indicesRowNumOnce = indicesNumPerLoop;
        if (runParams.indicesNumEachCore % runParams.indicesRowNumOnce != 0) {
          runParams.indicesRowNumLast = runParams.indicesNumEachCore % runParams.indicesRowNumOnce;
        }

        // the following parameters are not used in TILING_MODE_6
        runParams.rowNumOnceUb = paramsElemPerUb / runParams.paramsRow;

        runParams.innerLoopNum = runParams.indicesRowNumOnce / runParams.rowNumOnceUb;
        if (runParams.indicesRowNumOnce % runParams.rowNumOnceUb != 0) {
          runParams.rowNumOnceTailUb = runParams.indicesRowNumOnce % runParams.rowNumOnceUb;
        }

        runParams.innerLoopNumLast = runParams.indicesRowNumLast / runParams.rowNumOnceUb;
        if (runParams.indicesRowNumLast % runParams.rowNumOnceUb != 0) {
          runParams.rowNumLastTailUb = runParams.indicesRowNumLast % runParams.rowNumOnceUb;
        }
      } else {
        if (indicesTotal != 1) {
          tilingMode = TILING_MODE_7;  // one params row need tiling

          runParams.indicesLoopNum = runParams.indicesNumEachCore / halfUbIndicesNum;
          runParams.indicesRowNumOnce = halfUbIndicesNum;
          if (runParams.indicesNumEachCore % runParams.indicesRowNumOnce != 0) {
            runParams.indicesRowNumLast = runParams.indicesNumEachCore % runParams.indicesRowNumOnce;
          }

          runParams.oneRowLoop = runParams.paramsRow / halfUbParamsElem;
          runParams.oneRowTail = runParams.paramsRow % halfUbParamsElem;
          if (runParams.oneRowLoop > 0 && runParams.oneRowTail > 0 && runParams.oneRowTail < blockNum) {
            runParams.oneRowLoop = runParams.oneRowLoop - 1;
            runParams.oneRowTail = halfUbParamsElem + runParams.oneRowTail;
          }
        } else {
          tilingMode = TILING_MODE_9;  // indices shape = [1]

          int32_t ubParamsElem = (ubSize - UB_2K_SIZE) / paramsDSize;
          runParams.needCoreNum = coreNum;

          runParams.innerLoopNum = paramsSuffixList[0] / (ubParamsElem * runParams.needCoreNum);
          int32_t paramsTail = paramsSuffixList[0] % (ubParamsElem * runParams.needCoreNum);
          runParams.innerLoopNumLast = paramsTail / ubParamsElem;
          runParams.rowNumLastTailUb = paramsTail % ubParamsElem;

          if (runParams.innerLoopNum >= 1 && runParams.rowNumLastTailUb <= blockNum) {
            runParams.innerLoopNum = runParams.innerLoopNum - 1;
            runParams.rowNumLastTailUb = ubParamsElem + runParams.rowNumLastTailUb;
          }
        }
      }
    }
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  // set run tiling data
  runParams.tilingMode = tilingMode;
  SetRunningParams(runParams, runInfo);
  runInfo.AddTilingData(PARAMS_SUFFIX_INDEX);  // start index of the paramsSuffixList: 19
  for (int i = 0; i < LAST_DIM_MAX; i++) {
    if (i < indicesLastDim) {
      runInfo.AddTilingData(paramsSuffixList[i]);
    } else {
      runInfo.AddTilingData(0);
    }
  }

  PrintGatherNdrunParams(runParams);
  // block_dim, core num used in tik op
  runInfo.SetBlockDim(runParams.needCoreNum);

  OP_LOGD(opType.c_str(), "op[GatherNdTiling] tiling run success.");
  PROFILING_TILING_END();

  return true;
}

// register tiling interface of the GatherNd op
REGISTER_OP_TILING_V3_WITH_VECTOR(GatherNd, GatherNdTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
