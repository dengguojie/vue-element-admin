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
 * \file sparse_apply_ftrl_d.cpp
 * \brief dynamic SparseApplyFtrl op tiling
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

const int32_t BLOCK_SIZE = 32;
const int32_t VECTOR_SIZE = 256;
// The 4KB space of UB is used to store indices data
const int32_t UB_INDICES_SIZE = 4 * 1024;
const int32_t UB_2K_SIZE = 2 * 1024;

// one row size of var is 32B aligned
const int32_t TILING_MODE_1 = 1;
// one row size of var is smaller than 32B
const int32_t TILING_MODE_2 = 2;
// indices num is smaller than (coreNum/2), and one row size of var is 32B aligned and large than 1024 elements
const int32_t TILING_MODE_3 = 3;

bool CheckTensorShape(const std::string& opType, const TeOpParas& opParas, int32_t& varRowElem) {
  std::vector<int64_t> varShape = opParas.inputs[0].tensor[0].shape;
  std::vector<int64_t> accumShape = opParas.inputs[1].tensor[0].shape;
  std::vector<int64_t> linearShape = opParas.inputs[2].tensor[0].shape;
  std::vector<int64_t> gradShape = opParas.inputs[3].tensor[0].shape;
  std::vector<int64_t> indicesShape = opParas.inputs[4].tensor[0].shape;
  int32_t varDims = varShape.size();

  if (indicesShape[0] != gradShape[0]) {

    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [SparseApplyFtrlTiling] : grad shape[0] must be equal to indices shape[0]");
    return false;
  }

  for (int32_t i = 0; i < varDims; i++) {
    if (varShape[i] != accumShape[i] || varShape[i] != linearShape[i]) {

      VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [SparseApplyFtrlTiling] : accum and linear shape must be equal to var shape");
      return false;
    }
    if (i > 0) {
      if (varShape[i] != gradShape[i]) {

        VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [SparseApplyFtrlTiling] : grad shape is invalid");
        return false;
      }
      varRowElem *= varShape[i];
    }
  }

  return true;
}

bool GetCompileParameters(const std::string& opType, const nlohmann::json& opCompileInfoJson, int32_t& coreNum,
                          int32_t& ubSize, int32_t& indicesDSize) {
  using namespace nlohmann;

  const auto& allVars = opCompileInfoJson["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SparseApplyFtrlTiling", "GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int32_t>();

  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [SparseApplyFtrlTiling] : GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int32_t>();

  if (allVars.count("indices_dsize") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [GatherV2Tiling] : GetCompileParams, get indices_dsize error");
    return false;
  }
  indicesDSize = allVars["indices_dsize"].get<std::int32_t>();

  GELOGD("op [SparseApplyFtrlTiling] : GetCompileParams, coreNum[%d], ubSize[%d].", coreNum, ubSize);
  return true;
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool SparseApplyFtrlDTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                            OpRunInfo& runInfo) {
  GELOGI("op[%s] tiling running.", opType.c_str());
  if (op_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SparseApplyFtrlTiling: op_info json error.");
    return false;
  }
  if (opParas.inputs.empty() || opParas.inputs.size() < 5 || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty() || opParas.inputs[2].tensor.empty() || opParas.inputs[3].tensor.empty() ||
      opParas.inputs[4].tensor.empty()) {

    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SparseApplyFtrlTiling: input shape error.");
    return false;
  }

  int32_t varRowElem = 1;
  // check inputs shape
  bool ret = CheckTensorShape(opType, opParas, varRowElem);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op[parseApplyFtrlTiling] SparseApplyFtrlTiling: inputs shape are invalid.");
    return ret;
  }

  std::vector<int64_t> varShape = opParas.inputs[0].tensor[0].shape;
  std::vector<int64_t> indicesShape = opParas.inputs[4].tensor[0].shape;
  int32_t varRows = varShape[0];
  int32_t indicesNums = indicesShape[0];

  int32_t varDSize = 4;  // only support float32
  int32_t varElemBlock = BLOCK_SIZE / varDSize;
  int32_t varElemVector = VECTOR_SIZE / varDSize;

  // get compile info
  int32_t indicesDSize = 4;
  int32_t ubSize = 0;
  int32_t coreNum = 0;
  bool flag = GetCompileParameters(opType, op_info, coreNum, ubSize, indicesDSize);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op SparseApplyFtrlTiling: GetCompileParams error.");
    return false;
  }
  int32_t ubIndicesNum = UB_INDICES_SIZE / indicesDSize;
  int32_t remainUbSize = ubSize - UB_2K_SIZE - UB_INDICES_SIZE;
  int32_t onePartUbSize = remainUbSize / 6;
  int32_t onePartElem = onePartUbSize / varDSize;
  onePartElem = onePartElem - onePartElem % varElemVector;

  if (varRowElem > onePartElem) {

    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SparseApplyFtrlTiling: inputs var row elements is too large, is not support yet.");
    return false;
  }

  int32_t needCoreNum = coreNum;
  int32_t tailProcessCore = 0;
  int32_t indicesNumEachCore = 0;
  int32_t indicesNumRemaining = 0;
  int32_t tilingMode = TILING_MODE_1;

  int32_t numMultiRows = 32;  // must be 32 factor for align
  int32_t indicesStep = 0;

  int32_t partialFactor = 0;
  int32_t elemsPerCore = 0;
  int32_t elemsLastCore = 0;
  int32_t elemsCoreLoop = 0;
  int32_t elemsCoreRemain = 0;
  int32_t elemsLastCoreLoop = 0;
  int32_t elemsLastCoreRemain = 0;

  if (varRowElem < varElemBlock) {
    tilingMode = TILING_MODE_2;

    indicesNumEachCore = indicesNums;
    indicesNumRemaining = 0;

    if (varRows < numMultiRows) {
      numMultiRows = varRows;
    }
    needCoreNum = varRows / numMultiRows;
    if (needCoreNum > coreNum) {
      needCoreNum = coreNum;
    }
    if (needCoreNum <= 0) {
      needCoreNum = 1;
    }
    indicesStep = varRows / needCoreNum;
  } else if ((varRowElem >= varElemBlock) && (varRowElem % varElemBlock == 0)) {
    if (indicesNums * 2 < coreNum && varRowElem >= 1024) {  // 1024 = 32*32
      tilingMode = TILING_MODE_3;

      indicesNumEachCore = indicesNums;
      indicesNumRemaining = 0;

      needCoreNum = indicesNums;
      partialFactor = coreNum / needCoreNum;
      needCoreNum = needCoreNum * partialFactor;
      elemsPerCore = varRowElem / partialFactor;
      elemsLastCore = varRowElem - (partialFactor - 1) * elemsPerCore;

      elemsCoreLoop = elemsPerCore / onePartElem;
      elemsCoreRemain = elemsPerCore % onePartElem;
      elemsLastCoreLoop = elemsLastCore / onePartElem;
      elemsLastCoreRemain = elemsLastCore % onePartElem;
    } else {
      tilingMode = TILING_MODE_1;

      indicesNumEachCore = indicesNums / needCoreNum;
      indicesNumRemaining = indicesNums % needCoreNum;
      if (indicesNums <= needCoreNum) {
        needCoreNum = indicesNums;
        tailProcessCore = 0;
        indicesNumEachCore = 1;
        indicesNumRemaining = 0;
      }
    }
  } else {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op SparseApplyFtrlTiling: inputs var row elements is not 32B aligned.");
    return false;
  }

  // useless in TILING_MODE_3, because indices nums is smaller than core nums
  int32_t indicesLoopNum = indicesNumEachCore / ubIndicesNum;
  int32_t indicesNumsOnce = ubIndicesNum;
  int32_t indicesNumLast = 0;
  if (indicesNumEachCore % indicesNumsOnce != 0) {
    indicesNumLast = indicesNumEachCore % indicesNumsOnce;
  }
  GELOGD("op [SparseApplyFtrlTiling] : indicesNumEachCore=%d, varRowElem=%d, indicesNums=%d", indicesNumEachCore,
         varRowElem, indicesNums);

  // set tiling data
  ByteBufferPut(runInfo.tiling_data, tilingMode);
  ByteBufferPut(runInfo.tiling_data, needCoreNum);
  ByteBufferPut(runInfo.tiling_data, tailProcessCore);
  ByteBufferPut(runInfo.tiling_data, indicesNumEachCore);
  ByteBufferPut(runInfo.tiling_data, indicesNumRemaining);
  ByteBufferPut(runInfo.tiling_data, indicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, indicesNumLast);
  ByteBufferPut(runInfo.tiling_data, varRowElem);

  ByteBufferPut(runInfo.tiling_data, varRows);
  ByteBufferPut(runInfo.tiling_data, indicesStep);
  ByteBufferPut(runInfo.tiling_data, numMultiRows);

  ByteBufferPut(runInfo.tiling_data, partialFactor);
  ByteBufferPut(runInfo.tiling_data, elemsPerCore);
  ByteBufferPut(runInfo.tiling_data, elemsLastCore);
  ByteBufferPut(runInfo.tiling_data, elemsCoreLoop);
  ByteBufferPut(runInfo.tiling_data, elemsCoreRemain);
  ByteBufferPut(runInfo.tiling_data, elemsLastCoreLoop);
  ByteBufferPut(runInfo.tiling_data, elemsLastCoreRemain);

  // block_dim, core num used in tik op
  runInfo.block_dim = needCoreNum;
  // workspace, null for tik op
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  GELOGI("op[%s] tiling run success.", opType.c_str());

  return true;
}

// register tiling interface of the SparseApplyFtrlD op
REGISTER_OP_TILING_FUNC_BUFFERED(SparseApplyFtrlD, SparseApplyFtrlDTiling);

}  // namespace optiling
