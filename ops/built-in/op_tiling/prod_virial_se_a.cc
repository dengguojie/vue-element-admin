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
 * \file prod_virial_se_a.cc
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
constexpr int64_t INPUT_LENGTH = 5;
constexpr int64_t OUTPUT_LENGTH = 2;
constexpr int64_t NDESCRPT_FACTOR = 4;
constexpr int64_t RIJ_FACTOR = 3;
constexpr int64_t OUTPUT_VIRIAL_FACTOR = 9;
constexpr int64_t NNEI_UB = 256;
constexpr int64_t CUSTOM_AICORE_NUM = 8;
constexpr int64_t CUSTOM_VECTORCORE_NUM = 7;

struct ProdVirialSeAParams {
  int64_t nneiPerFrame;
  int64_t nall;
  int64_t repTimesOffset;
  int64_t neiRepTimes;
  int64_t preCoreNum;
  int64_t postCoreNum;
  int64_t neiRepTimesPreCore;
  int64_t neiRepTimesPostCore;
};

bool CheckProdVirialSeAParams(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opInfo) {
  OP_LOGD(opType.c_str(), "CheckProdVirialSeAParams begin");

  OP_TILING_CHECK(opParas.inputs.size() != INPUT_LENGTH,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The length of inputs should be 5"), return false);
  OP_TILING_CHECK(opParas.outputs.size() != OUTPUT_LENGTH,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The length of outputs should be 2"), return false);

  OP_TILING_CHECK(opParas.inputs[0].tensor.empty() || opParas.inputs[1].tensor.empty() ||
                      opParas.inputs[2].tensor.empty() || opParas.inputs[3].tensor.empty() ||
                      opParas.inputs[4].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Some of input tensors is empty"), return false);
  OP_TILING_CHECK(opParas.outputs[0].tensor.empty() || opParas.outputs[1].tensor.empty(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Some of output tensors is empty"), return false);

  OP_LOGD(opType.c_str(), "CheckProdVirialSeAParams run success");
  return true;
}

bool GetProdVirialSeACompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int64_t& coreNum,
                                   int64_t& splitCount, int64_t& splitIndex) {
  OP_LOGD(opType.c_str(), "GetProdVirialSeACompileParams begin");

  auto allVars = opCompileInfo["vars"];
  OP_TILING_CHECK(allVars.count("core_num") == 0, VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get core_num"),
                  return false);
  coreNum = allVars["core_num"].get<std::int64_t>();

  OP_TILING_CHECK(allVars.count("split_count") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get split_count"), return false);
  splitCount = allVars["split_count"].get<std::int64_t>();
  OP_TILING_CHECK(splitCount < 1, VECTOR_INNER_ERR_REPORT_TILIING(opType, "Split count should be positive value"),
                  return false);

  OP_TILING_CHECK(allVars.count("split_index") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get split_index."), return false);
  splitIndex = allVars["split_index"].get<std::int64_t>();
  OP_TILING_CHECK(splitIndex < 0 || splitIndex >= splitCount,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Split index should be positive and less than split count"),
                  return false);

  OP_LOGD(opType.c_str(), "GetProdVirialSeACompileParams run success");
  return true;
}

bool CheckProdVirialSeAShapeInfo(const std::string& opType, const TeOpParas& opParas, int64_t& nframes,
                                 int64_t& nneiPerFrame, int64_t& nall) {
  OP_LOGD(opType.c_str(), "CheckProdVirialSeAShapeInfo begin");

  std::vector<int64_t> netShape = opParas.inputs[0].tensor[0].shape;
  OP_TILING_CHECK(netShape.size() != 2, VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of net_deriv should be 2"),
                  return false);

  std::vector<int64_t> inShape = opParas.inputs[1].tensor[0].shape;
  OP_TILING_CHECK(inShape.size() != 2, VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of in_deriv should be 2"),
                  return false);

  std::vector<int64_t> rijShape = opParas.inputs[2].tensor[0].shape;
  OP_TILING_CHECK(rijShape.size() != 2, VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of rij should be 2"),
                  return false);

  std::vector<int64_t> nlistShape = opParas.inputs[3].tensor[0].shape;
  OP_TILING_CHECK(nlistShape.size() != 2, VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of nlist should be 2"),
                  return false);

  nframes = netShape[0];
  nneiPerFrame = nlistShape[1];
  OP_TILING_CHECK(
      inShape[0] != nframes || rijShape[0] != nframes || nlistShape[0] != nframes,
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape[0] of {net_deriv, in_deriv, rij, nlist} should be consistent"),
      return false);
  OP_TILING_CHECK(netShape[1] != nneiPerFrame * NDESCRPT_FACTOR,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape[1] of net_deriv should be nloc * nnei * 3"),
                  return false);
  OP_TILING_CHECK(inShape[1] != nneiPerFrame * NDESCRPT_FACTOR * RIJ_FACTOR,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape[1] of in_deriv should be nloc * nnei * 4 * 3"),
                  return false);
  OP_TILING_CHECK(rijShape[1] != nneiPerFrame * RIJ_FACTOR,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape[1] of rij should be nloc * nnei * 3"),
                  return false);

  std::vector<int64_t> virialShape = opParas.outputs[0].tensor[0].shape;
  OP_TILING_CHECK(virialShape.size() != 2 || virialShape[0] != nframes || virialShape[1] != OUTPUT_VIRIAL_FACTOR,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of virial should be {nframes, 9}"), return false);

  std::vector<int64_t> atomVirialShape = opParas.outputs[1].tensor[0].shape;
  OP_TILING_CHECK(atomVirialShape.size() != 2,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of atom_virial should be 2"), return false);
  int64_t avDim = atomVirialShape[1];
  OP_TILING_CHECK(atomVirialShape[0] != nframes || avDim < 1 || avDim % OUTPUT_VIRIAL_FACTOR != 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "The shape of atom_virial should be {nframes, nall * 9}"),
                  return false);
  nall = avDim / OUTPUT_VIRIAL_FACTOR;

  OP_LOGD(opType.c_str(), "CheckProdVirialSeAShapeInfo run success");
  return true;
}

void InitProdVirialSeAParams(ProdVirialSeAParams& params) {
  params.nneiPerFrame = 0;
  params.nall = 0;
  params.repTimesOffset = 0;
  params.neiRepTimes = 0;
  params.preCoreNum = 0;
  params.postCoreNum = 0;
  params.neiRepTimesPreCore = 0;
  params.neiRepTimesPostCore = 0;
}

void SetProdVirialSeAParams(const ProdVirialSeAParams& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.nneiPerFrame);
  ByteBufferPut(runInfo.tiling_data, runParams.nall);
  ByteBufferPut(runInfo.tiling_data, runParams.repTimesOffset);
  ByteBufferPut(runInfo.tiling_data, runParams.neiRepTimes);
  ByteBufferPut(runInfo.tiling_data, runParams.preCoreNum);
  ByteBufferPut(runInfo.tiling_data, runParams.postCoreNum);
  ByteBufferPut(runInfo.tiling_data, runParams.neiRepTimesPreCore);
  ByteBufferPut(runInfo.tiling_data, runParams.neiRepTimesPostCore);
}

void PrintProdVirialSeAParams(const std::string& opType, const ProdVirialSeAParams& runParams) {
  OP_LOGD(opType.c_str(), "nneiPerFrame=%d", runParams.nneiPerFrame);
  OP_LOGD(opType.c_str(), "nall=%d", runParams.nall);
  OP_LOGD(opType.c_str(), "repTimesOffset=%d", runParams.repTimesOffset);
  OP_LOGD(opType.c_str(), "neiRepTimes=%d", runParams.neiRepTimes);
  OP_LOGD(opType.c_str(), "preCoreNum=%d", runParams.preCoreNum);
  OP_LOGD(opType.c_str(), "postCoreNum=%d", runParams.postCoreNum);
  OP_LOGD(opType.c_str(), "neiRepTimesPreCore=%d", runParams.neiRepTimesPreCore);
  OP_LOGD(opType.c_str(), "neiRepTimesPostCore=%d", runParams.neiRepTimesPostCore);
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] opCompileInfo: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool ProdVirialSeATiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                         OpRunInfo& runInfo) {
  OP_LOGD(opType.c_str(), "ProdVirialSeATiling run begin");
  OP_TILING_CHECK(!CheckProdVirialSeAParams(opType, opParas, opCompileInfo),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to check input params"), return false);

  int64_t coreNum;
  int64_t splitCount;
  int64_t splitIndex;
  OP_TILING_CHECK(!GetProdVirialSeACompileParams(opType, opCompileInfo, coreNum, splitCount, splitIndex),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get compile params"), return false);

  int64_t nframes;
  int64_t nneiPerFrame;
  int64_t nall;
  OP_TILING_CHECK(!CheckProdVirialSeAShapeInfo(opType, opParas, nframes, nneiPerFrame, nall),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to check shape info"), return false);

  int64_t repTimesTotal = nframes;
  if (nneiPerFrame >= NNEI_UB) {
    repTimesTotal = (nframes * nneiPerFrame + NNEI_UB - 1) / NNEI_UB;
  }

  int64_t repTimesOffset = 0;
  int64_t repTimesFix = repTimesTotal;
  if (splitCount == 2) {  // Split by 8:7
    int64_t totalCoreNum = CUSTOM_AICORE_NUM + CUSTOM_VECTORCORE_NUM;
    int64_t repTimesSplit = (repTimesTotal + totalCoreNum - 1) / totalCoreNum;
    if (splitIndex == 0) {
      repTimesFix = repTimesSplit * CUSTOM_AICORE_NUM;
    } else {
      repTimesOffset = repTimesSplit * CUSTOM_AICORE_NUM;
      repTimesFix = repTimesTotal - repTimesOffset;
    }
  }

  ProdVirialSeAParams runParams;
  InitProdVirialSeAParams(runParams);
  runParams.nneiPerFrame = nneiPerFrame;
  runParams.nall = nall;
  runParams.repTimesOffset = repTimesOffset;
  runParams.neiRepTimes = repTimesFix;
  runParams.preCoreNum = repTimesFix % coreNum;
  runParams.postCoreNum = coreNum - runParams.preCoreNum;
  runParams.neiRepTimesPreCore = (repTimesFix + coreNum - 1) / coreNum;
  runParams.neiRepTimesPostCore = repTimesFix / coreNum;

  int64_t blockDim = repTimesFix < coreNum ? repTimesFix : coreNum;
  OP_LOGD(opType.c_str(), "blockDim=%d", blockDim);

  SetProdVirialSeAParams(runParams, runInfo);
  runInfo.block_dim = blockDim;
  PrintProdVirialSeAParams(opType, runParams);

  OP_LOGD(opType.c_str(), "ProdVirialSeATiling run success");
  return true;
}
// register tiling interface of the ProdVirialSeA op.
REGISTER_OP_TILING_FUNC_BUFFERED(ProdVirialSeA, ProdVirialSeATiling);
}  // namespace optiling
