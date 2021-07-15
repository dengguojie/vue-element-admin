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
 * \file gather_v2.cpp
 * \brief tiling function of op
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

const int64_t BLOCK_SIZE = 32;
const int64_t PARAMS_CACHED_UB = 100 * 1024;
const int64_t RESERVED_UB_SIZE = 6 * 1024;

// A. block tiling: indices tiling
// 1. one params row size is smaller than 32B
// params is not cache
const int64_t TILING_MODE_1 = 1;
// params is cache in UB
const int64_t TILING_MODE_4 = 4;
// params is cache in L1
const int64_t TILING_MODE_13 = 13;

// 2. one params row size is greater than or equal to 32B
// paramsRow is not 32B aligned
const int64_t TILING_MODE_2 = 2;
// the data of one params row can not store in half UB, need tiling
const int64_t TILING_MODE_5 = 5;

// 3. paramsRow is 32B aligned
// params is not cache in UB or L1
const int64_t TILING_MODE_3 = 3;
// params is cache in UB
const int64_t TILING_MODE_6 = 6;
// params is cache in L1
const int64_t TILING_MODE_7 = 7;

// B. block tiling: params_pre tiling
// 1. one params row size is smaller than 32B
// params is not cache
const int64_t TILING_MODE_8 = 8;
// params is cache in UB
const int64_t TILING_MODE_9 = 9;

// 2. paramsRow is 32B aligned
// params is not cache in UB or L1
const int64_t TILING_MODE_10 = 10;
// params is cache in UB
const int64_t TILING_MODE_11 = 11;
// params is cache in L1
const int64_t TILING_MODE_12 = 12;

struct GatherV2TilingParams {
  int64_t tilingMode;
  int64_t paramsPre;
  int64_t paramsAxis;
  int64_t paramsRow;
  int64_t indicesNum;
  int64_t cacheParams;
  int64_t need_core_num;
  int64_t tail_process_core;
  int64_t indices_num_each_core;
  int64_t indices_num_remaining;
  int64_t indices_loop_num;
  int64_t indices_row_num_once;
  int64_t indices_row_num_last;
  int64_t row_num_once_ub;
  int64_t row_num_once_tail_ub;
  int64_t inner_loop_num;
  int64_t row_num_last_ub;
  int64_t row_num_last_tail_ub;
  int64_t inner_loop_num_last;
  int64_t paramsTotal;
  int64_t oneRowLoop;
  int64_t oneRowTail;
  int64_t params_pre_each_core;
  int64_t params_pre_remaining;
};

void InitGatherV2Params(GatherV2TilingParams& params) {
  params.tilingMode = 0;
  params.paramsPre = 1;
  params.paramsAxis = 1;
  params.paramsRow = 1;
  params.indicesNum = 1;
  params.cacheParams = 0;
  params.need_core_num = 0;
  params.tail_process_core = 0;
  params.indices_num_each_core = 0;
  params.indices_num_remaining = 0;
  params.indices_loop_num = 0;
  params.indices_row_num_once = 0;
  params.indices_row_num_last = 0;
  params.row_num_once_ub = 0;
  params.row_num_once_tail_ub = 0;
  params.inner_loop_num = 0;
  params.row_num_last_ub = 0;
  params.row_num_last_tail_ub = 0;
  params.inner_loop_num_last = 0;
  params.paramsTotal = 0;
  params.oneRowLoop = 0;
  params.oneRowTail = 0;
  params.params_pre_each_core = 0;
  params.params_pre_remaining = 0;
}

void SetGatherV2Params(GatherV2TilingParams& Params, OpRunInfo& runInfo) {
  // set tiling data
  ByteBufferPut(runInfo.tiling_data, Params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, Params.paramsPre);
  ByteBufferPut(runInfo.tiling_data, Params.paramsAxis);
  ByteBufferPut(runInfo.tiling_data, Params.paramsRow);
  ByteBufferPut(runInfo.tiling_data, Params.indicesNum);
  ByteBufferPut(runInfo.tiling_data, Params.cacheParams);
  ByteBufferPut(runInfo.tiling_data, Params.need_core_num);
  ByteBufferPut(runInfo.tiling_data, Params.tail_process_core);
  ByteBufferPut(runInfo.tiling_data, Params.indices_num_each_core);
  ByteBufferPut(runInfo.tiling_data, Params.indices_num_remaining);
  ByteBufferPut(runInfo.tiling_data, Params.indices_loop_num);
  ByteBufferPut(runInfo.tiling_data, Params.indices_row_num_once);
  ByteBufferPut(runInfo.tiling_data, Params.indices_row_num_last);
  ByteBufferPut(runInfo.tiling_data, Params.row_num_once_ub);
  ByteBufferPut(runInfo.tiling_data, Params.row_num_once_tail_ub);
  ByteBufferPut(runInfo.tiling_data, Params.inner_loop_num);
  ByteBufferPut(runInfo.tiling_data, Params.row_num_last_ub);
  ByteBufferPut(runInfo.tiling_data, Params.row_num_last_tail_ub);
  ByteBufferPut(runInfo.tiling_data, Params.inner_loop_num_last);
  ByteBufferPut(runInfo.tiling_data, Params.paramsTotal);
  ByteBufferPut(runInfo.tiling_data, Params.oneRowLoop);
  ByteBufferPut(runInfo.tiling_data, Params.oneRowTail);
  ByteBufferPut(runInfo.tiling_data, Params.params_pre_each_core);
  ByteBufferPut(runInfo.tiling_data, Params.params_pre_remaining);
}

void PrintGatherV2Params(const GatherV2TilingParams& params) {
  GELOGD("op [GatherV2Tiling] : tilingMode=%d.", params.tilingMode);
  GELOGD("op [GatherV2Tiling] : paramsPre=%d.", params.paramsPre);
  GELOGD("op [GatherV2Tiling] : paramsAxis=%d.", params.paramsAxis);
  GELOGD("op [GatherV2Tiling] : paramsRow=%d.", params.paramsRow);
  GELOGD("op [GatherV2Tiling] : indicesNum=%d.", params.indicesNum);
  GELOGD("op [GatherV2Tiling] : cacheParams=%d.", params.cacheParams);
  GELOGD("op [GatherV2Tiling] : need_core_num=%d.", params.need_core_num);
  GELOGD("op [GatherV2Tiling] : tail_process_core=%d.", params.tail_process_core);
  GELOGD("op [GatherV2Tiling] : indices_num_each_core=%d.", params.indices_num_each_core);
  GELOGD("op [GatherV2Tiling] : indices_num_remaining=%d.", params.indices_num_remaining);
  GELOGD("op [GatherV2Tiling] : indices_loop_num=%d.", params.indices_loop_num);
  GELOGD("op [GatherV2Tiling] : indices_row_num_once=%d.", params.indices_row_num_once);
  GELOGD("op [GatherV2Tiling] : indices_row_num_last=%d.", params.indices_row_num_last);
  GELOGD("op [GatherV2Tiling] : row_num_once_ub=%d.", params.row_num_once_ub);
  GELOGD("op [GatherV2Tiling] : row_num_once_tail_ub=%d.", params.row_num_once_tail_ub);
  GELOGD("op [GatherV2Tiling] : inner_loop_num=%d.", params.inner_loop_num);
  GELOGD("op [GatherV2Tiling] : row_num_last_ub=%d.", params.row_num_last_ub);
  GELOGD("op [GatherV2Tiling] : row_num_last_tail_ub=%d.", params.row_num_last_tail_ub);
  GELOGD("op [GatherV2Tiling] : inner_loop_num_last=%d.", params.inner_loop_num_last);
  GELOGD("op [GatherV2Tiling] : paramsTotal=%d.", params.paramsTotal);
  GELOGD("op [GatherV2Tiling] : oneRowLoop=%d.", params.oneRowLoop);
  GELOGD("op [GatherV2Tiling] : oneRowTail=%d.", params.oneRowTail);
  GELOGD("op [GatherV2Tiling] : params_pre_each_core=%d.", params.params_pre_each_core);
  GELOGD("op [GatherV2Tiling] : params_pre_remaining=%d.", params.params_pre_remaining);
}

bool checkTensorShape(const std::string& opType, std::vector<int64_t> paramsShape, std::vector<int64_t> indicesShape,
                      std::vector<int64_t> indicesOriShape, std::vector<int64_t> yShape, int64_t axis) {
  int64_t paramsDims = paramsShape.size();
  int64_t indicesDims = indicesShape.size();
  int64_t indicesOriDims = indicesOriShape.size();
  int64_t yDims = yShape.size();

  std::vector<int64_t> outputShape;
  if (axis > 0) {
    for (int64_t i = 0; i < axis; i++) {
      outputShape.push_back(paramsShape[i]);
    }
  }
  if (indicesOriDims > 0) {
    for (int64_t i = 0; i < indicesDims; i++) {
      outputShape.push_back(indicesShape[i]);
    }
  }
  if (axis + 1 < paramsDims) {
    for (int64_t i = axis + 1; i < paramsDims; i++) {
      outputShape.push_back(paramsShape[i]);
    }
  }
  int64_t outputDims = outputShape.size();

  if (yDims != outputDims) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [GatherV2Tiling] : CheckTensorShape, y Shape is invalid.");
    return false;
  }

  for (int64_t i = 0; i < yDims; i++) {
    if (yShape[i] != outputShape[i]) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [GatherV2Tiling] : CheckTensorShape, y Shpae dim is invalid.");
      return false;
    }
  }

  return true;
}

bool GetV2CompileParams(const std::string& opType, const nlohmann::json& opCompileInfoJson, int64_t& coreNum,
                        int64_t& ubSize, int64_t& l1Size, int64_t& paramsDSize, int64_t& indicesDSize) {
  using namespace nlohmann;

  const auto& allVars = opCompileInfoJson["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [GatherV2Tiling] : GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();
  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [GatherV2Tiling] : GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int64_t>();
  if (allVars.count("l1_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [GatherV2Tiling] : GetCompileParams, get l1_size error");
    return false;
  }
  l1Size = allVars["l1_size"].get<std::int64_t>();
  if (allVars.count("params_dsize") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [GatherV2Tiling] : GetCompileParams, get params_dsize error");
    return false;
  }
  paramsDSize = allVars["params_dsize"].get<std::int64_t>();
  if (allVars.count("indices_dsize") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [GatherV2Tiling] : GetCompileParams, get indices_dsize error");
    return false;
  }
  indicesDSize = allVars["indices_dsize"].get<std::int64_t>();

  return true;
}

// compute tiling params for tiling_mode 8&9
bool BlockLessForParamsTiling(GatherV2TilingParams& runParams, int64_t& indicesNumPerLoop, int64_t& resUbSize,
                              int64_t& paramsDSize, int64_t& blockNum) {
  runParams.indices_loop_num = runParams.indices_num_each_core / indicesNumPerLoop;
  runParams.indices_row_num_once = indicesNumPerLoop;
  if (runParams.indices_num_each_core % runParams.indices_row_num_once != 0) {
    runParams.indices_row_num_last = runParams.indices_num_each_core % runParams.indices_row_num_once;
  }

  runParams.row_num_once_ub = resUbSize / (runParams.paramsRow * paramsDSize);
  if (int(runParams.row_num_once_ub % blockNum) != 0) {
    runParams.row_num_once_ub = int(runParams.row_num_once_ub / blockNum) * blockNum;
  }
  OP_TILING_CHECK((runParams.row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2", "Devide by row_num_once_ub[%ld] exception.",
                                                  runParams.row_num_once_ub),
                  return false);
  runParams.inner_loop_num = runParams.indices_row_num_once / runParams.row_num_once_ub;
  if (runParams.indices_row_num_once % runParams.row_num_once_ub != 0) {
    runParams.row_num_once_tail_ub = runParams.indices_row_num_once % runParams.row_num_once_ub;
  }
  if (runParams.inner_loop_num > 0 && runParams.row_num_once_tail_ub > 0
      && runParams.row_num_once_tail_ub * runParams.paramsRow < blockNum) {
    runParams.inner_loop_num = runParams.inner_loop_num - 1;
    runParams.row_num_once_tail_ub = runParams.row_num_once_tail_ub + runParams.row_num_once_ub;
  }

  runParams.row_num_last_ub = resUbSize / (runParams.paramsRow * paramsDSize);
  if (int(runParams.row_num_last_ub % blockNum) != 0) {
    runParams.row_num_last_ub = int(runParams.row_num_last_ub / blockNum) * blockNum;
  }
  OP_TILING_CHECK((runParams.row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2", "Devide by row_num_last_ub[%ld] exception.",
                                                  runParams.row_num_last_ub),
                  return false);
  runParams.inner_loop_num_last = runParams.indices_row_num_last / runParams.row_num_last_ub;
  if (runParams.indices_row_num_last % runParams.row_num_last_ub != 0) {
    runParams.row_num_last_tail_ub = runParams.indices_row_num_last % runParams.row_num_last_ub;
  }
  if (runParams.inner_loop_num_last > 0 && runParams.row_num_last_tail_ub > 0
      && runParams.row_num_last_tail_ub * runParams.paramsRow < blockNum) {
    runParams.inner_loop_num_last = runParams.inner_loop_num_last - 1;
    runParams.row_num_last_tail_ub = runParams.row_num_last_tail_ub + runParams.row_num_once_ub;
  }

  return true;
}

// compute tiling params for tiling_mode 10&11&12
bool BlockAlignForParamsTiling(GatherV2TilingParams& runParams, int64_t& indicesNumPerLoop,
                               int64_t& resUbSize, int64_t& paramsDSize) {
  runParams.indices_loop_num = runParams.indices_num_each_core / indicesNumPerLoop;
  runParams.indices_row_num_once = indicesNumPerLoop;
  if (runParams.indices_num_each_core % runParams.indices_row_num_once != 0) {
    runParams.indices_row_num_last = runParams.indices_num_each_core % runParams.indices_row_num_once;
  }

  runParams.row_num_once_ub = resUbSize / (runParams.paramsRow * paramsDSize);
  OP_TILING_CHECK((runParams.row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2", "Devide by row_num_once_ub[%ld] exception.",
                                                  runParams.row_num_once_ub),
                  return false);
  runParams.inner_loop_num = runParams.indices_row_num_once / runParams.row_num_once_ub;
  if (runParams.indices_row_num_once % runParams.row_num_once_ub != 0) {
    runParams.row_num_once_tail_ub = runParams.indices_row_num_once % runParams.row_num_once_ub;
  }

  runParams.row_num_last_ub = resUbSize / (runParams.paramsRow * paramsDSize);
  OP_TILING_CHECK((runParams.row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2", "Devide by row_num_last_ub[%ld] exception.",
                                                  runParams.row_num_last_ub),
                  return false);
  runParams.inner_loop_num_last = runParams.indices_row_num_last / runParams.row_num_last_ub;
  if (runParams.indices_row_num_last % runParams.row_num_last_ub != 0) {
    runParams.row_num_last_tail_ub = runParams.indices_row_num_last % runParams.row_num_last_ub;
  }

  return true;
}

// compute tiling params for tiling_mode 1&4&13
bool BlockLessForIndicesTiling(GatherV2TilingParams& runParams, int64_t& indicesNumPerLoop, int64_t& resUbSize,
                               int64_t& paramsDSize, int64_t& blockNum) {
  runParams.indices_loop_num = runParams.indices_num_each_core / indicesNumPerLoop;
  runParams.indices_row_num_once = indicesNumPerLoop;
  if (runParams.indices_num_each_core % runParams.indices_row_num_once != 0) {
    runParams.indices_row_num_last = runParams.indices_num_each_core % runParams.indices_row_num_once;
  }

  runParams.row_num_once_ub = resUbSize / (runParams.paramsRow * paramsDSize);
  if (int(runParams.row_num_once_ub % blockNum) != 0) {
    runParams.row_num_once_ub = int(runParams.row_num_once_ub / blockNum) * blockNum;
  }
  OP_TILING_CHECK((runParams.row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2", "Devide by row_num_once_ub[%ld] exception.",
                                                  runParams.row_num_once_ub),
                  return false);
  runParams.inner_loop_num = runParams.indices_row_num_once / runParams.row_num_once_ub;
  if (runParams.indices_row_num_once % runParams.row_num_once_ub != 0) {
    runParams.row_num_once_tail_ub = runParams.indices_row_num_once % runParams.row_num_once_ub;
  }
  if (runParams.inner_loop_num > 0 && runParams.row_num_once_tail_ub > 0 &&
      runParams.row_num_once_tail_ub * runParams.paramsRow < blockNum) {
    runParams.inner_loop_num = runParams.inner_loop_num - 1;
    runParams.row_num_once_tail_ub = runParams.row_num_once_tail_ub + runParams.row_num_once_ub;
  }

  runParams.row_num_last_ub = resUbSize / (runParams.paramsRow * paramsDSize);
  if (int(runParams.row_num_last_ub % blockNum) != 0) {
    runParams.row_num_last_ub = int(runParams.row_num_last_ub / blockNum) * blockNum;
  }
  OP_TILING_CHECK((runParams.row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2", "Devide by row_num_last_ub[%ld] exception.",
                                                  runParams.row_num_last_ub),
                  return false);
  runParams.inner_loop_num_last = runParams.indices_row_num_last / runParams.row_num_last_ub;
  if (runParams.indices_row_num_last % runParams.row_num_last_ub != 0) {
    runParams.row_num_last_tail_ub = runParams.indices_row_num_last % runParams.row_num_last_ub;
  }
  if (runParams.inner_loop_num_last > 0 && runParams.row_num_last_tail_ub > 0 &&
      runParams.row_num_last_tail_ub * runParams.paramsRow < blockNum) {
    runParams.inner_loop_num_last = runParams.inner_loop_num_last - 1;
    runParams.row_num_last_tail_ub = runParams.row_num_last_tail_ub + runParams.row_num_once_ub;
  }

  return true;
}

// compute tiling params for tiling_mode 3&6&7
bool BlockAlignForIndicesTiling(GatherV2TilingParams& runParams, int64_t& indicesNumPerLoop,
                                int64_t& resUbSize, int64_t& paramsDSize) {
  runParams.indices_loop_num = runParams.indices_num_each_core / indicesNumPerLoop;
  runParams.indices_row_num_once = indicesNumPerLoop;
  if (runParams.indices_num_each_core % runParams.indices_row_num_once != 0) {
    runParams.indices_row_num_last = runParams.indices_num_each_core % runParams.indices_row_num_once;
  }

  runParams.row_num_once_ub = resUbSize / (runParams.paramsRow * paramsDSize);
  OP_TILING_CHECK((runParams.row_num_once_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2", "Devide by row_num_once_ub[%ld] exception.",
                                                  runParams.row_num_once_ub),
                  return false);
  runParams.inner_loop_num = runParams.indices_row_num_once / runParams.row_num_once_ub;
  if (runParams.indices_row_num_once % runParams.row_num_once_ub != 0) {
    runParams.row_num_once_tail_ub = runParams.indices_row_num_once % runParams.row_num_once_ub;
  }

  runParams.row_num_last_ub = resUbSize / (runParams.paramsRow * paramsDSize);
  OP_TILING_CHECK((runParams.row_num_last_ub == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING("GatherV2", "Devide by row_num_last_ub[%ld] exception.",
                                                  runParams.row_num_last_ub),
                  return false);
  runParams.inner_loop_num_last = runParams.indices_row_num_last / runParams.row_num_last_ub;
  if (runParams.indices_row_num_last % runParams.row_num_last_ub != 0) {
    runParams.row_num_last_tail_ub = runParams.indices_row_num_last % runParams.row_num_last_ub;
  }

  return true;
}

void CalNeedCore(int64_t& needCore, int64_t& indicesEachCore, int64_t& indicesRemain,
                 int64_t& indicesNum, int64_t& paramsRow, int64_t& paramsDSize) {
  while (needCore > 1) {
    needCore = needCore / 2;
    indicesEachCore = indicesNum / needCore;
    indicesRemain = indicesNum % needCore;
    if (indicesEachCore * paramsRow * paramsDSize > BLOCK_SIZE) {
      break;
    }
  }
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool GatherV2Tiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                    OpRunInfo& runInfo) {
  GELOGI("op[%s] GatherV2Tiling running.", opType.c_str());
  using namespace ge;
  if (op_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op GatherV2Tiling: op_info json error.");
    return false;
  }
  if (opParas.inputs.empty() || opParas.inputs.size() < 3 || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty() || opParas.inputs[2].tensor.empty()) {

    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op GatherV2Tiling: input shape error.");
    return false;
  }
  if (opParas.outputs.empty() || opParas.outputs.size() < 1 || opParas.outputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op GatherV2Tiling: output shape error.");
    return false;
  }

  std::vector<int64_t> paramsShape = opParas.inputs[0].tensor[0].shape;
  std::vector<int64_t> indicesShape = opParas.inputs[1].tensor[0].shape;
  std::vector<int64_t> indicesOriShape = opParas.inputs[1].tensor[0].ori_shape;
  std::vector<int64_t> axisShape = opParas.inputs[2].tensor[0].shape;
  std::vector<int64_t> yShape = opParas.outputs[0].tensor[0].shape;

  if (opParas.const_inputs.find("axis") == opParas.const_inputs.end()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op GatherV2Tiling: axis not exists.");
    return false;
  }

  std::string axis_type = opParas.inputs[2].tensor[0].dtype;
  int64_t axis;
  if (axis_type == "int64") {
    const int64_t* axis_ptr = reinterpret_cast<const int64_t*>(std::get<0>(opParas.const_inputs.at("axis")));
    axis = *axis_ptr;
  } else if (axis_type == "int32") {
    const int32_t* axis_ptr = reinterpret_cast<const int32_t*>(std::get<0>(opParas.const_inputs.at("axis")));
    axis = *axis_ptr;
  } else {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op GatherV2Tiling: axis can only suppport int32 and int64.");
    return false;
  }
  GELOGD("op [GatherV2Tiling] : axis=%ld.", axis);

  // check inputs shape
  int64_t paramsDims = paramsShape.size();
  int64_t indicesDims = indicesShape.size();
  if (paramsDims <= 0 || indicesDims <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GatherV2Tiling: paramsDims or indicesDims is 0.");
    return false;
  }
  if (axis < -paramsDims || axis >= paramsDims) {

    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op GatherV2Tiling: axis is invalid.");
    return false;
  }
  if (axis < 0) {
    axis += paramsDims;
  }

  bool ret = checkTensorShape(opType, paramsShape, indicesShape, indicesOriShape, yShape, axis);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op GatherV2Tiling: [checkTensorShape] failed.");
    return ret;
  }

  // get compile info
  int64_t ubSize = 0;
  int64_t l1Size = 0;
  int64_t coreNum = 0;
  int64_t paramsDSize = 0;
  int64_t indicesDSize = 0;
  bool flag = GetV2CompileParams(opType, op_info, coreNum, ubSize, l1Size, paramsDSize, indicesDSize);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GatherV2Tiling: GetV2CompileParams error.");
    return false;
  }

  int64_t availableUbSize = ubSize - 2 * 1024;  // reserved 2K
  int64_t halfUbSize = availableUbSize / 2;
  int64_t blockNum = BLOCK_SIZE / paramsDSize;

  GatherV2TilingParams runParams;
  InitGatherV2Params(runParams);

  // params shape convert to 3D:[paramsPre, paramsAxis, paramsRow]
  // indices shape convert to 1D:[indicesNum]
  // output tensor, y shape convert to:[paramsPre, indicesNum, paramsRow]
  if (axis == 0) {
    runParams.paramsPre = 1;
  } else {
    for (int i = 0; i < axis; i++) {
      runParams.paramsPre *= paramsShape[i];
    }
  }
  runParams.paramsAxis = paramsShape[axis];
  if (axis + 1 < paramsDims) {
    for (int i = axis + 1; i < paramsDims; i++) {
      runParams.paramsRow *= paramsShape[i];
    }
  } else {
    runParams.paramsRow = 1;
  }

  runParams.paramsTotal = std::accumulate(paramsShape.begin(), paramsShape.end(), 1, std::multiplies<int64_t>());
  int64_t paramsTotalCeil = (runParams.paramsTotal + blockNum - 1) / blockNum * blockNum;

  int64_t paramsRowCeil = (runParams.paramsRow + blockNum - 1) / blockNum * blockNum;

  for (int i = 0; i < indicesDims; i++) {
    runParams.indicesNum *= indicesShape[i];
  }

  int64_t resUbSize = halfUbSize;  // store params row data
  int64_t halfUbIndicesElem = halfUbSize / indicesDSize;
  int64_t indicesNumPerLoop = halfUbIndicesElem;
  int64_t halfRemainUbSize = (availableUbSize - PARAMS_CACHED_UB) / 2;
  int64_t halfRemainParamsElem = halfRemainUbSize / paramsDSize;
  int64_t halfUbParamsElem = halfUbSize / paramsDSize;

  // the data of the formula gained from actual tests
  // set a gate value for tiling_mode_7 to optimized some data_move processes
  float mode_7_gate_value = 56.5 - 0.012 * runParams.paramsTotal / 1024;

  if (runParams.paramsPre >= coreNum && paramsRowCeil <= halfUbParamsElem &&
      (runParams.paramsRow * paramsDSize < BLOCK_SIZE || runParams.paramsRow * paramsDSize % BLOCK_SIZE == 0)) {
    // block tiling: params_pre tiling
    runParams.need_core_num = coreNum;
    runParams.tail_process_core = 0;
    runParams.params_pre_each_core = runParams.paramsPre / runParams.need_core_num;
    runParams.params_pre_remaining = runParams.paramsPre % runParams.need_core_num;
    runParams.indices_num_each_core = runParams.indicesNum;

    if (runParams.indices_num_each_core * runParams.paramsRow * paramsDSize <= BLOCK_SIZE) {
      runParams.need_core_num = 1;
      runParams.tail_process_core = 0;
      runParams.params_pre_each_core = runParams.paramsPre;
      runParams.params_pre_remaining = 0;
    }

    if (runParams.paramsRow * paramsDSize < BLOCK_SIZE) {
      if (paramsTotalCeil <= PARAMS_CACHED_UB / paramsDSize) {
        runParams.tilingMode = TILING_MODE_8;
      } else {
        runParams.tilingMode = TILING_MODE_9;
      }

      if (runParams.tilingMode == TILING_MODE_8) {
        indicesNumPerLoop = halfRemainUbSize / indicesDSize;
        resUbSize = halfRemainUbSize;
      }

      if (!BlockLessForParamsTiling(runParams, indicesNumPerLoop, resUbSize, paramsDSize, blockNum)) {
        return false;
      }
    } else {
      if (paramsTotalCeil <= PARAMS_CACHED_UB / paramsDSize && paramsRowCeil <= halfRemainParamsElem) {
        runParams.tilingMode = TILING_MODE_10;
      } else if (paramsTotalCeil <= l1Size / paramsDSize) {
        runParams.tilingMode = TILING_MODE_11;
      } else {
        runParams.tilingMode = TILING_MODE_12;
      }

      if (runParams.tilingMode == TILING_MODE_10) {
        indicesNumPerLoop = halfRemainUbSize / indicesDSize;
        resUbSize = halfRemainUbSize;
      }

      if (!BlockAlignForParamsTiling(runParams, indicesNumPerLoop, resUbSize, paramsDSize)) {
        return false;
      }
    }
  } else {
    // block tiling: indices tiling
    runParams.need_core_num = coreNum;
    runParams.tail_process_core = 0;
    runParams.indices_num_each_core = runParams.indicesNum / runParams.need_core_num;
    runParams.indices_num_remaining = runParams.indicesNum % runParams.need_core_num;
    if (runParams.indicesNum <= runParams.need_core_num) {
      runParams.need_core_num = runParams.indicesNum;
      runParams.tail_process_core = 0;
      runParams.indices_num_each_core = 1;
      runParams.indices_num_remaining = 0;
    }

    // one params row size is smaller than 32B
    if (runParams.paramsRow * paramsDSize < BLOCK_SIZE) {
      if (paramsTotalCeil <= PARAMS_CACHED_UB / paramsDSize) {
        runParams.tilingMode = TILING_MODE_4;
      } else if (paramsTotalCeil <= l1Size / paramsDSize) {
        runParams.tilingMode = TILING_MODE_13;
      } else {
        runParams.tilingMode = TILING_MODE_1;
      }

      if ((runParams.paramsRow < BLOCK_SIZE) &&
          runParams.indices_num_each_core * runParams.paramsRow * paramsDSize <= BLOCK_SIZE) {
          CalNeedCore(runParams.need_core_num, runParams.indices_num_each_core,
                      runParams.indices_num_remaining, runParams.indicesNum,
                      runParams.paramsRow, paramsDSize);
      }

      if (runParams.tilingMode == TILING_MODE_4) {
        indicesNumPerLoop = halfRemainUbSize / indicesDSize;
        resUbSize = halfRemainUbSize;
      }

      if (!BlockLessForIndicesTiling(runParams, indicesNumPerLoop, resUbSize, paramsDSize, blockNum)) {
        return false;
      }
    } else {                                            // one params row size is greater than or equal to 32B
      if (paramsRowCeil <= halfUbParamsElem) {
        if (runParams.paramsRow * paramsDSize % BLOCK_SIZE != 0) {  // not 32B aligned
          runParams.tilingMode = TILING_MODE_2;

          runParams.indices_loop_num = runParams.indices_num_each_core / halfUbIndicesElem;
          runParams.indices_row_num_once = halfUbIndicesElem;
          if (runParams.indices_num_each_core % runParams.indices_row_num_once != 0) {
            runParams.indices_row_num_last = runParams.indices_num_each_core % runParams.indices_row_num_once;
          }
        } else {  // 32B aligned
          if (paramsTotalCeil <= PARAMS_CACHED_UB / paramsDSize && paramsRowCeil <= halfRemainParamsElem) {
            runParams.tilingMode = TILING_MODE_6;
          } else if (paramsTotalCeil <= l1Size / paramsDSize &&  runParams.indicesNum > mode_7_gate_value) {
            runParams.tilingMode = TILING_MODE_7;
          } else {
            runParams.tilingMode = TILING_MODE_3;
          }

          if (runParams.tilingMode == TILING_MODE_6) {
            indicesNumPerLoop = halfRemainUbSize / indicesDSize;
            resUbSize = halfRemainUbSize;
          }

          if (!BlockAlignForIndicesTiling(runParams, indicesNumPerLoop, resUbSize, paramsDSize)) {
            return false;
          }
        }
      } else {
        runParams.tilingMode = TILING_MODE_5;  // one params row need tiling

        runParams.indices_loop_num = runParams.indices_num_each_core / halfUbIndicesElem;
        runParams.indices_row_num_once = indicesNumPerLoop;
        if (runParams.indices_num_each_core % runParams.indices_row_num_once != 0) {
          runParams.indices_row_num_last = runParams.indices_num_each_core % runParams.indices_row_num_once;
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

  SetGatherV2Params(runParams, runInfo);
  PrintGatherV2Params(runParams);

  // block_dim, core num used in tik op
  runInfo.block_dim = runParams.need_core_num;
  // workspace, null for tik op
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  GELOGI("op[%s] tiling run success.", opType.c_str());

  return true;
}

// register tiling interface of the GatherV2 op.
REGISTER_OP_TILING_FUNC_BUFFERED(GatherV2, GatherV2Tiling);

}  // namespace optiling
