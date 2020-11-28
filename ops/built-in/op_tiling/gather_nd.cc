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
 * \file gather_nd.cpp
 * \brief tiling function of op
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"

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

bool CheckTensorShape(const std::string& opType, std::vector<int64_t> paramsShape, std::vector<int64_t> indicesShape,
                      std::vector<int64_t> yShape, int32_t indicesLastDim) {
  int32_t paramsDims = paramsShape.size();
  int32_t indicesDims = indicesShape.size();
  int32_t yDims = yShape.size();
  int32_t calcOutputDims = 0;

  if (indicesLastDim == 0 && indicesDims == 1) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices", "the last dim of indices can not be equal to 0 or 1");
    OP_LOGE(opType.c_str(), "op [GatherNdTiling] : CheckTensorShape, indices.shape is invalid.");
    return false;
  }
  // output shape is: indicesShape[:-1] + paramsShape[indicesShape[-1]:]
  std::vector<int64_t> outputShape;
  for (int32_t i = 0; i < indicesDims - 1; i++) {
    if (indicesShape[i] <= 0) {
      ge::OpsOneInputShapeErrReport(opType.c_str(), "indices", "the shape i of indices must be large than 0");
      OP_LOGE(opType.c_str(), "op [GatherNdTiling] : CheckTensorShape, indices.shape[i] must be > 0");
      return false;
    }
    outputShape.push_back(indicesShape[i]);
  }
  for (int32_t i = indicesLastDim; i < paramsDims; i++) {
    outputShape.push_back(paramsShape[i]);
  }
  calcOutputDims = outputShape.size();

  if (yDims != calcOutputDims) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "y", "the dim of y must be equal to the dim of output");
    OP_LOGE(opType.c_str(), "op [GatherNdTiling] : CheckTensorShape, y Shape dim is invalid");
    return false;
  }

  for (int32_t i = 0; i < yDims; i++) {
    if (yShape[i] != outputShape[i]) {
      ge::OpsOneInputShapeErrReport(opType.c_str(), "y", "the shape of y must be equal to the shape of output");
      OP_LOGE(opType.c_str(), "op [GatherNdTiling] : CheckTensorShape, y Shape is invalid");
      return false;
    }
  }

  return true;
}

bool GetCompileParams(const std::string& opType, const nlohmann::json& opCompileInfoJson, int32_t& coreNum,
                      int32_t& ubSize, int32_t& l1Size, int32_t& paramsDSize, int32_t& indicesDSize) {
  using namespace nlohmann;

  const auto& allVars = opCompileInfoJson["vars"];
  if (allVars.count("core_num") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "core_num");
    OP_LOGE(opType.c_str(), "op [GatherNdTiling] : GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int32_t>();

  if (allVars.count("ub_size") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "ub_size");
    OP_LOGE(opType.c_str(), "op [GatherNdTiling] : GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int32_t>();

  if (allVars.count("l1_size") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "l1_size");
    OP_LOGE(opType.c_str(), "op [GatherNdTiling] : GetCompileParams, get l1_size error");
    return false;
  }
  l1Size = allVars["l1_size"].get<std::int32_t>();

  if (allVars.count("params_dsize") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "params_dsize");
    OP_LOGE(opType.c_str(), "op [GatherNdTiling] : GetCompileParams, get params_dsize error");
    return false;
  }
  paramsDSize = allVars["params_dsize"].get<std::int32_t>();

  if (allVars.count("indices_dsize") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "indices_dsize");
    OP_LOGE(opType.c_str(), "op [GatherNdTiling] : GetCompileParams, get indices_dsize error");
    return false;
  }
  indicesDSize = allVars["indices_dsize"].get<std::int32_t>();

  GELOGD("op [GatherNdTiling] : GetCompileParams, coreNum[%d], ubSize[%d], l1Size[%d].", coreNum, ubSize, l1Size);
  return true;
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

void SetRunningParams(const GatherNdParam& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.tilingMode);

  ByteBufferPut(runInfo.tiling_data, runParams.needCoreNum);
  ByteBufferPut(runInfo.tiling_data, runParams.tailProcessCore);
  ByteBufferPut(runInfo.tiling_data, runParams.indicesNumEachCore);
  ByteBufferPut(runInfo.tiling_data, runParams.indicesNumRemaining);
  ByteBufferPut(runInfo.tiling_data, runParams.indicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, runParams.indicesRowNumOnce);
  ByteBufferPut(runInfo.tiling_data, runParams.indicesRowNumLast);
  ByteBufferPut(runInfo.tiling_data, runParams.rowNumOnceUb);
  ByteBufferPut(runInfo.tiling_data, runParams.rowNumOnceTailUb);
  ByteBufferPut(runInfo.tiling_data, runParams.innerLoopNum);
  ByteBufferPut(runInfo.tiling_data, runParams.rowNumLastTailUb);
  ByteBufferPut(runInfo.tiling_data, runParams.innerLoopNumLast);

  ByteBufferPut(runInfo.tiling_data, runParams.paramsRow);
  ByteBufferPut(runInfo.tiling_data, runParams.indicesLastDim);
  ByteBufferPut(runInfo.tiling_data, runParams.paramsTotal);
  ByteBufferPut(runInfo.tiling_data, runParams.oneRowLoop);
  ByteBufferPut(runInfo.tiling_data, runParams.oneRowTail);
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool GatherNdTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                    OpRunInfo& runInfo) {
  GELOGI("op[%s] tiling running.", opType.c_str());
  if (op_info == nullptr) {
    OP_LOGE(opType.c_str(), "op GatherNdTiling: op_info json error.");
    return false;
  }
  if (opParas.inputs.empty() || opParas.inputs.size() < 2 || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "x or indices",
                                  "The length of inputs is less than 2 or the inputs is empty");
    OP_LOGE(opType.c_str(), "op GatherNdTiling: input shape error.");
    return false;
  }
  if (opParas.outputs.empty() || opParas.outputs.size() < 1 || opParas.outputs[0].tensor.empty()) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "y", "The length of outputs is less than 1 or the outputs is empty");
    OP_LOGE(opType.c_str(), "op GatherNdTiling: output shape error.");
    return false;
  }

  std::vector<int64_t> paramsShape = opParas.inputs[0].tensor[0].shape;
  std::vector<int64_t> indicesShape = opParas.inputs[1].tensor[0].shape;
  std::vector<int64_t> yShape = opParas.outputs[0].tensor[0].shape;
  int32_t paramsDims = paramsShape.size();
  int32_t indicesDims = indicesShape.size();
  if (indicesDims < 1) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices", "the dim of indices is less than 1.");
    OP_LOGE(opType.c_str(), "op GatherNdTiling: indices dim is invalid.");
    return false;
  }

  int32_t indicesLastDim = indicesShape[indicesDims - 1];
  // indices.shape[-1] must be <= params.rank, and shape only support 1D ~ 8D
  if (indicesLastDim > paramsDims || indicesLastDim > LAST_DIM_MAX || indicesLastDim < 0) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices",
                                  "the last dim of indices is more than the dim of x, "
                                  "or the last dim of indices is greater than 8 or less than 0");
    OP_LOGE(opType.c_str(), "op GatherNdTiling: the last dim of indices shape is invalid.");
    return false;
  }

  // check inputs shape
  bool ret = CheckTensorShape(opType, paramsShape, indicesShape, yShape, indicesLastDim);
  if (!ret) {
    OP_LOGE(opType.c_str(), "op GatherNdTiling: inputs shape are invalid.");
    return ret;
  }

  // get compile info
  int32_t ubSize = 0;
  int32_t l1Size = 0;
  int32_t coreNum = 0;
  int32_t paramsDSize = 0;
  int32_t indicesDSize = 0;
  bool flag = GetCompileParams(opType, op_info, coreNum, ubSize, l1Size, paramsDSize, indicesDSize);
  if (!flag) {
    OP_LOGE(opType.c_str(), "op GatherNdTiling: GetCompileParams error.");
    return false;
  }

  int32_t availableUbSize = ubSize - RESERVED_UB_SIZE;
  int32_t halfUbSize = availableUbSize / 2;

  int32_t paramsTotal = std::accumulate(paramsShape.begin(), paramsShape.end(), 1, std::multiplies<int64_t>());
  int32_t paramsTotalTmp = paramsTotal;
  // e.g. paramsShape:(2, 100, 4, 3) => paramsSuffixList:[100*4*3, 4*3, 3, 1]
  std::vector<int32_t> paramsSuffixList;
  for (int i = 0; i < indicesLastDim; i++) {
    paramsTotalTmp = paramsTotalTmp / paramsShape[i];
    paramsSuffixList.push_back(paramsTotalTmp);
  }

  // init running parameters
  GatherNdParam runParams;
  InitRunningParams(runParams);
  runParams.indicesLastDim = indicesLastDim;
  runParams.paramsTotal = paramsTotal;

  runParams.paramsRow = 1;
  for (int i = indicesLastDim; i < paramsDims; i++) {
    runParams.paramsRow *= paramsShape[i];
  }
  GELOGD("op [GatherNdTiling] : paramsDims=%d, indicesDims=%d, indicesLastDim=%d, paramsRow=%d", paramsDims,
         indicesDims, indicesLastDim, runParams.paramsRow);
  int64_t indicesPrefix = std::accumulate(indicesShape.begin(), indicesShape.end() - 1, 1,
                                          std::multiplies<int64_t>());  // the number of indices

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

    if (runParams.paramsRow < blockNum) {
      if (runParams.indicesNumEachCore * runParams.paramsRow <= blockNum) {
        runParams.needCoreNum = 1;
        runParams.tailProcessCore = 0;
        runParams.indicesNumEachCore = indicesPrefix;
        runParams.indicesNumRemaining = 0;
      }

      if (paramsTotalCeil <= PARAMS_CACHED_UB / paramsDSize) {
        tilingMode = TILING_MODE_2;
      } else {
        tilingMode = TILING_MODE_1;
      }

      if (tilingMode == TILING_MODE_2) {
        indicesNumPerLoop = halfRemainIndicesElem / indicesLastDim;
        paramsElemPerUb = halfRemainParamsElem;
        GELOGD("op [GatherNdTiling] : indicesNumPerLoop=%d, paramsElemPerUb=%d.", indicesNumPerLoop, paramsElemPerUb);
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
          GELOGD("op [GatherNdTiling] : indicesNumPerLoop=%d, paramsElemPerUb=%d.", indicesNumPerLoop, paramsElemPerUb);
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
      }
    }
  }

  // set run tiling data
  GELOGD("op [GatherNdTiling] : tilingMode=%d.", tilingMode);
  runParams.tilingMode = tilingMode;
  SetRunningParams(runParams, runInfo);
  ByteBufferPut(runInfo.tiling_data, PARAMS_SUFFIX_INDEX);  // start index of the paramsSuffixList: 19
  for (int i = 0; i < LAST_DIM_MAX; i++) {
    if (i < indicesLastDim) {
      ByteBufferPut(runInfo.tiling_data, paramsSuffixList[i]);
    } else {
      ByteBufferPut(runInfo.tiling_data, 0);
    }
  }

  // block_dim, core num used in tik op
  runInfo.block_dim = runParams.needCoreNum;
  // workspace, null for tik op
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  GELOGI("op[%s] tiling run success.", opType.c_str());

  return true;
}

// register tiling interface of the GatherNd op
REGISTER_OP_TILING_FUNC_BUFFERED(GatherNd, GatherNdTiling);

}  // namespace optiling
