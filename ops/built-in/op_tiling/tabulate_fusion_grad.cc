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
 * \file tabulate_fusion_grad.cc
 * \brief tiling function of op
 */
#include <string>
#include <nlohmann/json.hpp>
#include "../op_proto/util/error_util.h"
#include "graph/utils/op_desc_utils.h"
#include "error_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"
#include "op_log.h"

namespace optiling {
constexpr int64_t INPUT_LENGTH = 6;
constexpr int64_t OUTPUT_LENGTH = 2;

struct TabulateFusionGradParams {
  int64_t nloc;
  int64_t nnei;
  int64_t lastLayerSize;

  int64_t nlocOffset;
  int64_t nlocSplit;

  int64_t highCoreNum;
  int64_t lowCoreNum;
  int64_t locPerHighCore;
  int64_t locPerLowCore;
};

static bool CheckTabulateFusionGradParams(const std::string& opType, const TeOpParas& opParas,
                                          const nlohmann::json& opInfo) {
  OP_LOGD(opType.c_str(), "CheckTabulateFusionGradParams begin.");

  OP_TILING_CHECK(opParas.inputs.size() != INPUT_LENGTH,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The length of inputs should be 6"), return false);
  OP_TILING_CHECK(opParas.outputs.size() != OUTPUT_LENGTH,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The length of outputs should be 2"), return false);

  OP_TILING_CHECK(opParas.inputs[0].tensor.empty() || opParas.inputs[1].tensor.empty() ||
                  opParas.inputs[2].tensor.empty() || opParas.inputs[3].tensor.empty() ||
                  opParas.inputs[4].tensor.empty() || opParas.inputs[5].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Some of input tensors is empty"), return false);
  OP_TILING_CHECK(opParas.outputs[0].tensor.empty() || opParas.outputs[1].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Some of output tensors is empty"), return false);

  OP_LOGD(opType.c_str(), "CheckTabulateFusionGradParams run success.");
  return true;
}

static bool GetTabulateFusionGradCompileParams(const std::string& opType,
                                               const nlohmann::json& opCompileInfo, int64_t& coreNum,
                                               int64_t& splitCount, int64_t& splitIndex) {
  OP_LOGD(opType.c_str(), "GetTabulateFusionGradCompileParams begin.");

  auto allVars = opCompileInfo["vars"];
  OP_TILING_CHECK(allVars.count("core_num") == 0, VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get core_num."),
                  return false);
  coreNum = allVars["core_num"].get<std::int64_t>();

  OP_TILING_CHECK(allVars.count("split_count") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get split_count."), return false);
  splitCount = allVars["split_count"].get<std::int64_t>();
  OP_TILING_CHECK(splitCount < 1 || splitCount > 2,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Split count should be 1 or 2."),
                  return false);

  OP_TILING_CHECK(allVars.count("split_index") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get split_index."), return false);
  splitIndex = allVars["split_index"].get<std::int64_t>();
  OP_TILING_CHECK(splitIndex < 0 || splitIndex >= splitCount,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Split index should be positive and less than split count."),
                  return false);

  OP_LOGD(opType.c_str(), "GetTabulateFusionGradCompileParams run success.");
  return true;
}

static bool CheckTabulateFusionGradShapeInfo(const std::string& opType, const TeOpParas& opParas) {
  OP_LOGD(opType.c_str(), "CheckTabulateFusionGradShapeInfo begin.");

  std::vector<int64_t> descriptorShape = opParas.inputs[5].tensor[0].shape;
  OP_TILING_CHECK(descriptorShape.size() != 3, VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The dims(3 vs %lu) of descriptor is incorrect", descriptorShape.size()),
                  return false);
  OP_TILING_CHECK(descriptorShape[1] != 4 || descriptorShape[2] < 1, VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The shape[1](4 vs %ld) or shape[2](%ld > 0) of descriptor is incorrect",
                  descriptorShape[1], descriptorShape[2]),
                  return false);

  std::vector<int64_t> tableShape = opParas.inputs[0].tensor[0].shape;
  OP_TILING_CHECK(tableShape.size() != 2, VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The dims(2 vs %lu) of table is incorrect", tableShape.size()),
                  return false);
  OP_TILING_CHECK(tableShape[1] != 6 * descriptorShape[2], VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The shape[1](%ld vs %ld) of table is incorrect",
                  6 * descriptorShape[2], tableShape[1]),
                  return false);

  std::vector<int64_t> tableInfoShape = opParas.inputs[1].tensor[0].shape;
  OP_TILING_CHECK(tableInfoShape.size() != 1, VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The dims(1 vs %lu) of tableInfo is incorrect", tableInfoShape.size()),
                  return false);
  OP_TILING_CHECK(tableInfoShape[0] < 5, VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The shape[0](%ld >= 5) of tableInfo is incorrect", tableInfoShape[0]),
                  return false);

  std::vector<int64_t> emXShape = opParas.inputs[2].tensor[0].shape;
  OP_TILING_CHECK(emXShape.size() != 2, VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The dims(2 vs %lu) of emX is incorrect", emXShape.size()),
                  return false);

  std::vector<int64_t> emShape = opParas.inputs[3].tensor[0].shape;
  OP_TILING_CHECK(emShape.size() != 3, VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The dims(3 vs %lu) of em is incorrect", emShape.size()),
                  return false);
  OP_TILING_CHECK(emShape[2] != 4, VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The shape[2](%ld = 4) of em is incorrect", emShape[2]),
                  return false);

  std::vector<int64_t> dyShape = opParas.inputs[4].tensor[0].shape;
  OP_TILING_CHECK(dyShape.size() != 3, VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The dims(3 vs %lu) of dy is incorrect", dyShape.size()),
                  return false);
  OP_TILING_CHECK(dyShape[1] != 4 || dyShape[2] < 1, VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The shape[1](4 vs %ld) or shape[2](%ld > 0) of dy is incorrect", dyShape[1], dyShape[2]),
                  return false);

  std::vector<int64_t> dyDemXShape = opParas.outputs[0].tensor[0].shape;
  OP_TILING_CHECK(dyDemXShape.size() != emXShape.size(), VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The dims(%lu vs %lu) of dyDemX is incorrect", dyDemXShape.size(), emXShape.size()),
                  return false);
  OP_TILING_CHECK(dyDemXShape[1] != emXShape[1], VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The shape[1](%ld vs %ld) of dyDemx is incorrect", dyDemXShape[1], emXShape[1]),
                  return false);

  std::vector<int64_t> dyDemShape = opParas.outputs[1].tensor[0].shape;
  OP_TILING_CHECK(dyDemShape.size() != emShape.size(), VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The dims(%lu vs %lu) of dyDem is incorrect", dyDemShape.size(), emShape.size()),
                  return false);
  OP_TILING_CHECK(dyDemShape[1] != emShape[1] || dyDemShape[2] != emShape[2],
                  VECTOR_INNER_ERR_REPORT_TILIING(opType,
                  "The shape[1](%ld vs %ld) or shape[2](%ld vs %ld) of dyDem incorrect",
                  dyDemShape[1], emShape[1], dyDemShape[2], emShape[2]),
                  return false);

  OP_LOGD(opType.c_str(), "CheckTabulateFusionGradShapeInfo end.");
  return true;
}

static bool SetTabulateFusionGradRunParams(const std::string& opType, const TeOpParas& opParas,
                                           const int64_t& coreNum, const int64_t& splitCount,
                                           const int64_t& splitIndex, TabulateFusionGradParams& runParams) {
  OP_LOGD(opType.c_str(), "GetTabulateFusionGradRunParams begin.");

  std::vector<int64_t> emShape = opParas.inputs[3].tensor[0].shape;
  runParams.nloc = emShape[0];
  runParams.nnei = emShape[1];

  std::vector<int64_t> descriptorShape = opParas.inputs[5].tensor[0].shape;
  runParams.lastLayerSize = descriptorShape[2];

  int64_t nlocOffset = 0;
  int64_t nlocSplit = runParams.nloc;

  if (splitCount == 2) {
    if (splitIndex == 0) {
      nlocOffset = 0;
      nlocSplit = (runParams.nloc + 1) / 2;
    } else {
      nlocOffset = (runParams.nloc + 1) / 2;
      nlocSplit = runParams.nloc - nlocOffset;
    }
  }

  runParams.highCoreNum = nlocSplit % coreNum;
  runParams.lowCoreNum = coreNum - runParams.highCoreNum;
  runParams.locPerHighCore = (nlocSplit + coreNum - 1) / coreNum;
  runParams.locPerLowCore = nlocSplit / coreNum;

  runParams.nlocOffset = nlocOffset;
  runParams.nlocSplit = nlocSplit;

  OP_LOGD(opType.c_str(), "GetTabulateFusionGradRunParams end.");
  return true;
}

static void SetTabulateFusionGradRunInfo(const TabulateFusionGradParams& runParams, const int64_t& coreNum,
                                         OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.nloc);
  ByteBufferPut(runInfo.tiling_data, runParams.nnei);
  ByteBufferPut(runInfo.tiling_data, runParams.lastLayerSize);
  ByteBufferPut(runInfo.tiling_data, runParams.nlocOffset);
  ByteBufferPut(runInfo.tiling_data, runParams.nlocSplit);

  ByteBufferPut(runInfo.tiling_data, runParams.highCoreNum);
  ByteBufferPut(runInfo.tiling_data, runParams.lowCoreNum);
  ByteBufferPut(runInfo.tiling_data, runParams.locPerHighCore);
  ByteBufferPut(runInfo.tiling_data, runParams.locPerLowCore);

  runInfo.block_dim = runParams.nlocSplit < coreNum ? runParams.nlocSplit : coreNum;
}

static void PrintTabulateFusionGradRunParams(const std::string& opType,
                                             const TabulateFusionGradParams& runParams) {
  OP_LOGD(opType.c_str(), "nloc=%lld.", runParams.nloc);
  OP_LOGD(opType.c_str(), "nnei=%lld.", runParams.nnei);
  OP_LOGD(opType.c_str(), "lastLayerSize=%lld.", runParams.lastLayerSize);
  OP_LOGD(opType.c_str(), "nlocOffset=%lld.", runParams.nlocOffset);
  OP_LOGD(opType.c_str(), "nlocSplit=%lld.", runParams.nlocSplit);
  OP_LOGD(opType.c_str(), "highCoreNum=%lld.", runParams.highCoreNum);
  OP_LOGD(opType.c_str(), "lowCoreNum=%lld.", runParams.lowCoreNum);
  OP_LOGD(opType.c_str(), "locPerHighCore=%lld.", runParams.locPerHighCore);
  OP_LOGD(opType.c_str(), "locPerLowCore=%lld.", runParams.locPerLowCore);
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] opCompileInfo: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool TabulateFusionGradTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                              OpRunInfo& runInfo) {
  bool res = false;

  OP_LOGD(opType.c_str(), "Tiling run begin.");

  res = CheckTabulateFusionGradParams(opType, opParas, opCompileInfo);
  OP_TILING_CHECK(!res, VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to check input params"), return false);

  int64_t coreNum;
  int64_t splitCount;
  int64_t splitIndex;

  res = GetTabulateFusionGradCompileParams(opType, opCompileInfo, coreNum, splitCount, splitIndex);
  OP_TILING_CHECK(!res, VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get compile params"), return false);

  res = CheckTabulateFusionGradShapeInfo(opType, opParas);
  OP_TILING_CHECK(!res, VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to check shape info"), return false);

  TabulateFusionGradParams runParams = { 0 };
  res = SetTabulateFusionGradRunParams(opType, opParas, coreNum, splitCount, splitIndex, runParams);
  OP_TILING_CHECK(!res, VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to set run param"), return false);
  PrintTabulateFusionGradRunParams(opType, runParams);

  SetTabulateFusionGradRunInfo(runParams, coreNum, runInfo);
  OP_LOGD(opType.c_str(), "blockDim=%lld.", runInfo.block_dim);

  OP_LOGD(opType.c_str(), "Tiling run end.");
  return true;
}
// register tiling interface of the TabulateFusionGrad op.
REGISTER_OP_TILING_FUNC_BUFFERED(TabulateFusionGrad, TabulateFusionGradTiling);
}  // namespace optiling
