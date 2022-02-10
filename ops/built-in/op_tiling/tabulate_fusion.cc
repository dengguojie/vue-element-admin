/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file tabulate_fusion.cc
 * \brief
 */
#include <string>
#include <nlohmann/json.hpp>
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "error_log.h"
#include "../op_proto/util/error_util.h"
#include "op_tiling_util.h"

namespace optiling {
using namespace ge;

constexpr int64_t NUM_1 = 1;
constexpr int64_t NUM_2 = 2;
constexpr int64_t NUM_3 = 3;
constexpr int64_t NUM_4 = 4;
constexpr int64_t NUM_5 = 5;
constexpr int64_t NUM_6 = 6;
constexpr int64_t NUM_8 = 8;
constexpr int64_t NUM_15 = 15;
constexpr int64_t NUM_64 = 64;
constexpr int64_t NUM_128 = 128;
constexpr int64_t INDEX_2 = 2;
constexpr int64_t INDEX_3 = 3;

struct TabulateFusionTilingParams {
  int64_t needCoreNum;
  // nloc offset of aicore or vectorcore engine
  int64_t nlocEngineOffset;
  int64_t nnei;
  int64_t nlocOneCore;
  int64_t nlocLastCore;

  int64_t nlocPerLoop;
  // nloc loops for pre core
  int64_t preCoreLoops;
  int64_t preCoreNlocTail;
  // nloc loops for last core
  int64_t lastCoreLoops;
  int64_t lastCoreNlocTail;
};

struct TabulateFusionCompileParams {
  int64_t coreNum;
  int64_t lastLayerSize;
  int64_t onePortionElems;
  int64_t splitCount;
  int64_t splitIndex;
};

void InitTabulateFusionRunningParams(TabulateFusionTilingParams& params) {
  params.needCoreNum = 0;
  params.nlocEngineOffset = 0;
  params.nnei = 0;
  params.nlocOneCore = 0;
  params.nlocLastCore = 0;

  params.nlocPerLoop = 0;
  params.preCoreLoops = 0;
  params.preCoreNlocTail = 0;
  params.lastCoreLoops = 0;
  params.lastCoreNlocTail = 0;
}

bool CheckInOutSize(const std::string& opType, const TeOpParas& opParas) {
  OP_LOGD(opType.c_str(), "CheckInOutSize begin.");

  OP_TILING_CHECK(opParas.inputs.size() != NUM_4,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The length of inputs should be 4"), return false);
  OP_TILING_CHECK(opParas.outputs.size() != NUM_1,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The length of outputs should be 1"), return false);

  OP_TILING_CHECK(opParas.inputs[0].tensor.empty() || opParas.inputs[1].tensor.empty() ||
                      opParas.inputs[INDEX_2].tensor.empty() || opParas.inputs[INDEX_3].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Some of input tensors is empty"), return false);
  OP_TILING_CHECK(opParas.outputs[0].tensor.empty() || opParas.outputs[1].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Some of output tensors is empty"), return false);

  OP_LOGD(opType.c_str(), "CheckInOutSize run success.");
  return true;
}

bool CheckShapesInfo(const std::string& opType, const TeOpParas& opParas, int64_t lastLayerSize,
                     int64_t& nloc, int64_t& nnei) {
  OP_LOGD(opType.c_str(), "CheckShapesInfo begin");

  std::vector<int64_t> tableShape = opParas.inputs[0].tensor[0].shape;
  OP_TILING_CHECK(tableShape.size() != NUM_2,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of table should be 2"), return false);

  std::vector<int64_t> tableInfoShape = opParas.inputs[1].tensor[0].shape;
  OP_TILING_CHECK(tableInfoShape.size() != NUM_1,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of table_info should be 1"), return false);

  std::vector<int64_t> emXShape = opParas.inputs[INDEX_2].tensor[0].shape;
  OP_TILING_CHECK(emXShape.size() != NUM_2, VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of em_x should be 2"),
                  return false);

  std::vector<int64_t> emShape = opParas.inputs[INDEX_3].tensor[0].shape;
  OP_TILING_CHECK(emShape.size() != NUM_3, VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of em should be 3"),
                  return false);

  nloc = emShape[0];
  nnei = emShape[1];
  OP_LOGI(opType.c_str(), "CheckShapesInfo  nloc=%ld, nnei=%ld.", nloc, nnei);
  OP_TILING_CHECK((nloc <= 0) || (nnei <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The dim 0 and 1 of em should be greater than 0"),
                  return false);

  int64_t lastLayerSizeAlign = (lastLayerSize + NUM_64 - 1) / NUM_64 * NUM_64;
  OP_TILING_CHECK(tableShape[1] != lastLayerSizeAlign * NUM_6,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The dim 1 of table is invalid"),
                  return false);
  OP_TILING_CHECK(tableInfoShape[0] < NUM_5,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The dim 0 of table_info should not be less than 5"),
                  return false);
  OP_TILING_CHECK((emXShape[0] * emXShape[1] != nloc * nnei) || (emShape[INDEX_2] != NUM_4),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of em_x or em is invalid"), return false);

  std::vector<int64_t> descriptorShape = opParas.outputs[0].tensor[0].shape;
  OP_TILING_CHECK(descriptorShape.size() != NUM_3 ||
                      descriptorShape[1] != NUM_4 || descriptorShape[INDEX_2] != lastLayerSize,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of output should be (nloc, 4, last_layer_size)"),
                  return false);

  OP_LOGD(opType.c_str(), "CheckShapesInfo run success.");
  return true;
}

bool GetTabulateFusionCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo,
                                    TabulateFusionCompileParams& compileParams) {
  using namespace nlohmann;
  OP_LOGD(opType.c_str(), "GetTabulateFusionCompileParams begin");
  auto allVars = opCompileInfo["vars"];
  OP_TILING_CHECK(allVars.count("core_num") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get core_num."),
                  return false);
  compileParams.coreNum = allVars["core_num"].get<std::int64_t>();

  OP_TILING_CHECK(allVars.count("last_layer_size") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get last_layer_size."),
                  return false);
  compileParams.lastLayerSize = allVars["last_layer_size"].get<std::int64_t>();

  OP_TILING_CHECK(allVars.count("one_portion_ub_elems") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get one_portion_ub_elems."),
                  return false);
  compileParams.onePortionElems = allVars["one_portion_ub_elems"].get<std::int64_t>();

  OP_TILING_CHECK(allVars.count("split_count") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get split_count."),
                  return false);
  compileParams.splitCount = allVars["split_count"].get<std::int64_t>();

  OP_TILING_CHECK(allVars.count("split_index") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get split_index."),
                  return false);
  compileParams.splitIndex = allVars["split_index"].get<std::int64_t>();

  OP_LOGD(opType.c_str(), "GetTabulateFusionCompileParams run success.");
  return true;
}

void PrintTabulateFusionRunningParams(const std::string& opType, const TabulateFusionTilingParams& runParams) {
  OP_LOGI(opType.c_str(), "needCoreNum=%ld.", runParams.needCoreNum);
  OP_LOGI(opType.c_str(), "nlocEngineOffset=%ld.", runParams.nlocEngineOffset);
  OP_LOGI(opType.c_str(), "nnei=%ld.", runParams.nnei);
  OP_LOGI(opType.c_str(), "nlocOneCore=%ld.", runParams.nlocOneCore);
  OP_LOGI(opType.c_str(), "nlocLastCore=%ld.", runParams.nlocLastCore);
  OP_LOGI(opType.c_str(), "nlocPerLoop=%ld.", runParams.nlocPerLoop);
  OP_LOGI(opType.c_str(), "preCoreLoops=%ld.", runParams.preCoreLoops);
  OP_LOGI(opType.c_str(), "preCoreNlocTail=%ld.", runParams.preCoreNlocTail);
  OP_LOGI(opType.c_str(), "lastCoreLoops=%ld.", runParams.lastCoreLoops);
  OP_LOGI(opType.c_str(), "lastCoreNlocTail=%ld.", runParams.lastCoreNlocTail);
}

void SetTabulateFusionRuningParams(const TabulateFusionTilingParams& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.needCoreNum);
  ByteBufferPut(runInfo.tiling_data, params.nlocEngineOffset);
  ByteBufferPut(runInfo.tiling_data, params.nnei);
  ByteBufferPut(runInfo.tiling_data, params.nlocOneCore);
  ByteBufferPut(runInfo.tiling_data, params.nlocLastCore);
  ByteBufferPut(runInfo.tiling_data, params.nlocPerLoop);
  ByteBufferPut(runInfo.tiling_data, params.preCoreLoops);
  ByteBufferPut(runInfo.tiling_data, params.preCoreNlocTail);
  ByteBufferPut(runInfo.tiling_data, params.lastCoreLoops);
  ByteBufferPut(runInfo.tiling_data, params.lastCoreNlocTail);
}

bool CalTabulateFusionRunningParams(const std::string& opType, TabulateFusionTilingParams& runParams,
                                    int64_t nloc, int64_t nnei, TabulateFusionCompileParams& compileParam) {
  OP_LOGI(opType.c_str(), "CalTabulateFusionRunningParams begin");
  runParams.nlocEngineOffset = 0;
  int64_t nlocForEngine = nloc;
  // enable vector core
  if (compileParam.splitCount == NUM_2) {
    int64_t baseValue = nloc / NUM_15;
    int64_t ceilValue = baseValue * NUM_8;
    if (nloc % NUM_15 != 0) {
      ceilValue += (nloc % NUM_15);
    }

    if (compileParam.splitIndex == 0) {
      runParams.nlocEngineOffset = 0;
      nlocForEngine = ceilValue;
    } else {
      runParams.nlocEngineOffset = ceilValue;
      nlocForEngine = nloc - ceilValue;
    }
  }
  OP_LOGI(opType.c_str(), "CalTabulateFusionRunningParams, splitIndex=%ld, nlocForEngine=%ld, nlocEngineOffset=%ld",
          compileParam.splitIndex, nlocForEngine, runParams.nlocEngineOffset);

  int64_t nlocOneCore = (nlocForEngine + compileParam.coreNum - 1) / compileParam.coreNum;
  int64_t actCoreNum = nlocForEngine / nlocOneCore;
  int64_t nlocLastCore = nlocForEngine % nlocOneCore;
  if (nlocLastCore != 0) {
    actCoreNum = actCoreNum + 1;
  } else {
    nlocLastCore = nlocOneCore;
  }
  runParams.needCoreNum = actCoreNum;
  runParams.nnei = nnei;
  runParams.nlocOneCore = nlocOneCore;
  runParams.nlocLastCore = nlocLastCore;
  OP_LOGD(opType.c_str(), "actCoreNum=%ld, nlocOneCore=%ld, nlocLastCore=%ld", actCoreNum, nlocOneCore, nlocLastCore);

  OP_TILING_CHECK(compileParam.onePortionElems < nnei,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "nnei is too large and is not supported yet."),
                  return false);
  int64_t nlocPerLoop = compileParam.onePortionElems / nnei;
  if (nlocPerLoop > NUM_8) {
    nlocPerLoop = (nlocPerLoop / NUM_8) * NUM_8;
  }
  if (nlocPerLoop > NUM_128) {
    nlocPerLoop = NUM_128;
  }
  runParams.nlocPerLoop = nlocPerLoop;
  // loop times of one pre core
  runParams.preCoreLoops = nlocOneCore / nlocPerLoop;
  runParams.preCoreNlocTail = nlocOneCore - runParams.preCoreLoops * nlocPerLoop;
  // loop times of last core
  runParams.lastCoreLoops = nlocLastCore / nlocPerLoop;
  runParams.lastCoreNlocTail = nlocLastCore - runParams.lastCoreLoops * nlocPerLoop;

  OP_LOGI(opType.c_str(), "CalTabulateFusionRunningParams run success");
  return true;
}

bool TabulateFusionTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                          OpRunInfo& runInfo) {
  OP_LOGI(opType.c_str(), "TabulateFusionTiling run begin");
  OP_TILING_CHECK(!CheckInOutSize(opType, opParas),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to check input and output size"), return false);

  // get compile info
  TabulateFusionCompileParams compileParam;
  OP_TILING_CHECK(!GetTabulateFusionCompileParams(opType, opCompileInfo, compileParam) ||
                  (compileParam.coreNum == 0 || compileParam.lastLayerSize == 0 || compileParam.onePortionElems == 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get compile params"), return false);
  OP_TILING_CHECK(compileParam.splitCount < NUM_1 || compileParam.splitCount > NUM_2 || compileParam.splitIndex < 0 ||
                      compileParam.splitCount <= compileParam.splitIndex,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to check split info"), return false);

  int64_t nloc;
  int64_t nnei;
  OP_TILING_CHECK(!CheckShapesInfo(opType, opParas, compileParam.lastLayerSize, nloc, nnei),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to check shapes info"), return false);

  TabulateFusionTilingParams runParams;
  InitTabulateFusionRunningParams(runParams);
  OP_TILING_CHECK(!CalTabulateFusionRunningParams(opType, runParams, nloc, nnei, compileParam),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "CalTabulateFusionRunningParams failed"), return false);

  SetTabulateFusionRuningParams(runParams, runInfo);
  runInfo.block_dim = runParams.needCoreNum;
  PrintTabulateFusionRunningParams(opType, runParams);
  OP_LOGI(opType.c_str(), "TabulateFusionTiling run success.");

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(TabulateFusion, TabulateFusionTiling);
}  // namespace optiling
