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

#include <nlohmann/json.hpp>
#include <string>
#include <algorithm>
#include <vector>
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace {
  constexpr int64_t INPUT_SIZE = 4;
  constexpr int64_t INDEX_NLIST = 2;
  constexpr int64_t SPLIT_NUM_2 = 2;
}

namespace optiling {
struct ProdForceSeATilingParams {
  int64_t nloc;
  int64_t nall;
  int64_t coreLoopUnit;
  int64_t coreLoopLeft;
  int64_t coreOffset;
  int64_t nframes;
  int64_t coreNumsUsed;
};

void ProdForceSeAWriteTilingParams(const ProdForceSeATilingParams& params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, params.nloc);
  ByteBufferPut(run_info.tiling_data, params.nall);
  ByteBufferPut(run_info.tiling_data, params.coreLoopUnit);
  ByteBufferPut(run_info.tiling_data, params.coreLoopLeft);
  ByteBufferPut(run_info.tiling_data, params.coreOffset);
  ByteBufferPut(run_info.tiling_data, params.nframes);
  ByteBufferPut(run_info.tiling_data, params.coreNumsUsed);
}

void ProdForceSeAPrintTilingParams(const std::string& opType, const ProdForceSeATilingParams& params) {
  OP_LOGD("ProdForceSeA", "op [ProdForceSeA] : params.nloc=%d", params.nloc);
  OP_LOGD("ProdForceSeA", "op [ProdForceSeA] : params.nall=%d", params.nall);
  OP_LOGD("ProdForceSeA", "op [ProdForceSeA] : params.coreLoopUnit=%d", params.coreLoopUnit);
  OP_LOGD("ProdForceSeA", "op [ProdForceSeA] : params.coreLoopLeft=%d", params.coreLoopLeft);
  OP_LOGD("ProdForceSeA", "op [ProdForceSeA] : params.coreOffset=%d", params.coreOffset);
  OP_LOGD("ProdForceSeA", "op [ProdForceSeA] : params.nframes=%d", params.nframes);
  OP_LOGD("ProdForceSeA", "op [ProdForceSeA] : params.coreNumsUsed=%d", params.coreNumsUsed);
}

int64_t ProdForceSeACeilDiv(int64_t dividend, int64_t divisor) {
  return (dividend + divisor - 1) / divisor;
}

static const int64_t AI_CORE_NUM = 8;
static const int64_t VECTOR_CORE_NUM = 7;
static const int64_t NLIST_SHAPE = 2;
static const int64_t FORCE_SHAPE = 3;
static const int64_t SPLITE_NUMBER = 2;
bool ProdForceSeATiling(const std::string& opType, const TeOpParas& op_paras,
                        const nlohmann::json& op_compile_info_json,
                        OpRunInfo& run_info) {
  OP_LOGI("========================ProdForceSeATiling running====================");
  if (op_paras.inputs.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op_paras.inputs cannot be empty");
    return false;
  }
  if (op_paras.inputs.size() < INPUT_SIZE) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op_paras.inputs.size() < 4.");
    return false;
  }
  if (op_paras.outputs.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op_paras.outputs cannot be empty");
    return false;
  }
  if (op_paras.inputs[INDEX_NLIST].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "nlist tensor shape cannot be empty");
    return false;
  }
  std::vector<int64_t> nlistShape = op_paras.inputs[INDEX_NLIST].tensor[0].shape;
  if (nlistShape.size() < NLIST_SHAPE) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op_paras.inputs.size() < 4.");
    return false;
  }
  std::vector<int64_t> forceShape = op_paras.outputs[0].tensor[0].shape;
  if (forceShape.size() < FORCE_SHAPE) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op_paras.inputs.size() < 4.");
    return false;
  }
  const auto& allVars = op_compile_info_json["vars"];
  if (allVars.count("core_nums") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get core_nums fail.");
    return false;
  }
  int64_t coreNums = allVars["core_nums"].get<std::int64_t>();
  if (coreNums == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, coreNums is 0.");
    return false;
  }
  int64_t nASel = allVars["n_a_sel"].get<std::int64_t>();
  int64_t nRSel = allVars["n_r_sel"].get<std::int64_t>();
  int64_t splitCount = allVars["split_count"].get<std::int64_t>();
  if (splitCount < 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "splitCount is less than 1.");
    return false;
  }
  int64_t splitIndex = allVars["split_index"].get<std::int64_t>();
  if (splitIndex < 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "splitIndex is less than 0.");
    return false;
  }
  int64_t nnei = nASel + nRSel;
  int64_t nloc = nlistShape[1] / nnei;
  int64_t nframes = nlistShape[0];
  int64_t nall = forceShape[1];
  int64_t coreLoopLeft = 0;
  int64_t coreNumsUsed = 0;
  int64_t coreLoopUnit = 0;
  int64_t totalCoreNum = AI_CORE_NUM + VECTOR_CORE_NUM;
  int64_t aicNums = (nloc * AI_CORE_NUM) / totalCoreNum;
  int64_t vecNums = nloc - aicNums;
  int64_t nlocPartNums = nloc;
  if (splitCount == SPLITE_NUMBER) {
    nlocPartNums = (splitIndex == 0) ? aicNums : vecNums;
  }
  coreLoopUnit = ProdForceSeACeilDiv(nlocPartNums, coreNums);
  if (coreLoopUnit == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "coreLoopUnit is 0.");
    return false;
  }
  coreNumsUsed = ProdForceSeACeilDiv(nlocPartNums, coreLoopUnit);
  coreLoopLeft = nlocPartNums % coreLoopUnit;
  int64_t coreOffset = (splitIndex == 0) ? 0 : aicNums;
  ProdForceSeATilingParams params{nloc, nall, coreLoopUnit, coreLoopLeft, coreOffset, nframes, coreNumsUsed};
  run_info.block_dim = coreNums;
  ProdForceSeAWriteTilingParams(params, run_info);
  ProdForceSeAPrintTilingParams(opType, params);
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED(ProdForceSeA, ProdForceSeATiling);
}
