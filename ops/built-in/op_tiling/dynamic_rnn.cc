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
 * \file dynamic_rnn.cpp
 * \brief tiling function of op
 */
#include <string>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"

namespace optiling {
const int DEFAULT_SHAPE_LIST_SIZE = 3;
const int DEFAULT_INDEX_TWO = 2;
const int DEFAULT_RETURN = -2;
const int DEFAULT_PARAS_INPUT_SIZE = 2;
const int DEFAULT_XSHAPE_SIZE = 3;
const int DEFAULT_CHEQUE_INDEX = -2;
const int DEFAULT_BLOCK_DIM = 32;

struct DynamicRnnParam {
  int32_t sequenceLength;
  int32_t dynamicrnnbatch;
  int32_t chequeIndex;
};

void InitRunningParams(DynamicRnnParam& params) {
  params.sequenceLength = 0;
  params.dynamicrnnbatch = 0;
  params.chequeIndex = -1;
}

void SetRunningParams(const DynamicRnnParam& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.sequenceLength);
  ByteBufferPut(runInfo.tiling_data, runParams.dynamicrnnbatch);
  ByteBufferPut(runInfo.tiling_data, runParams.chequeIndex);
}

/*
 * @brief: print function of op
 * @param [in] opType: opType of the op
 * @param [in] params: tilling params
 * @return void: void
 */
void PrintTilingParams(const std::string& opType, const DynamicRnnParam& params) {
  OP_LOGD(opType.c_str(), "sequenceLength=%lld", params.sequenceLength);
  OP_LOGD(opType.c_str(), "dynamicrnnbatch=%lld", params.dynamicrnnbatch);
  OP_LOGD(opType.c_str(), "chequeIndex=%lld", params.chequeIndex);
}

// return tiling_indextiling_index
int32_t GetLibItem(const std::string& opType, const nlohmann::json& opCompileInfoJson, std::vector<int64_t> xShape) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfoJson["vars"];
  std::vector<std::vector<int64_t>> tune_shape_list =
      allVars.at("tune_shape_list").get<std::vector<std::vector<int64_t>>>();
  for (uint64_t i = 0; i < tune_shape_list.size(); i++) {
    if (tune_shape_list[i].size() != DEFAULT_SHAPE_LIST_SIZE) {
      OP_LOGE(opType.c_str(), "tune_shape_list's size is illegal. it's = %lld", tune_shape_list[i].size());
      return DEFAULT_RETURN;
    }
    if ((tune_shape_list[i][0] == -1) && (tune_shape_list[i][1] == -1)) {
      OP_LOGD(opType.c_str(), "matched floor schedule, the matched schedule = %lld", tune_shape_list[i][DEFAULT_INDEX_TWO]);
      int32_t res = (int32_t)tune_shape_list[i][DEFAULT_INDEX_TWO];
      return res;
    }
    if ((tune_shape_list[i][0] == xShape[0]) && (((tune_shape_list[i][1] + 15) / 16) == xShape[DEFAULT_INDEX_TWO])) {
      OP_LOGD(opType.c_str(), "The matched schedule = %lld", tune_shape_list[i][DEFAULT_INDEX_TWO]);
      int32_t res = (int32_t)tune_shape_list[i][DEFAULT_INDEX_TWO];
      return res;
    }
  }
  OP_LOGE(opType.c_str(), "No matching schedule is found.");
  return DEFAULT_RETURN;
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool DynamicRnnTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                      OpRunInfo& runInfo) {
  OP_LOGI(opType.c_str(), "tiling running.");
  if (op_info == nullptr) {
    OP_LOGE(opType.c_str(), "op DynamicRnnTiling: op_info json error.");
    return false;
  }
  if (opParas.inputs.empty() || opParas.inputs.size() < DEFAULT_PARAS_INPUT_SIZE || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty()) {
    OP_LOGE(opType.c_str(), "op DynamicRnnTiling: input shape error.");
    return false;
  }
  std::vector<int64_t> xShape = opParas.inputs[0].tensor[0].shape;
  if (xShape.size() < DEFAULT_XSHAPE_SIZE) {
    OP_LOGE(opType.c_str(), "op DynamicRnnTiling: inputs shape are invalid.");
    return false;
  }

  if (!(op_info.contains("vars")) || !(op_info.at("vars").contains("tune_shape_list"))) {
    OP_LOGE(opType.c_str(), "DynamicRnnTiling get tune_shape_list error.");
    return false;
  }

  // init running parameters
  DynamicRnnParam runParams;
  InitRunningParams(runParams);

  // set run tiling data
  int32_t sequenceLength = xShape[0];
  int32_t dynamicrnnbatch = xShape[DEFAULT_INDEX_TWO];
  int32_t chequeIndex = GetLibItem(opType, op_info, xShape);

  if (chequeIndex == DEFAULT_CHEQUE_INDEX) {
    OP_LOGE(opType.c_str(), "DynamicRnnTiling has no matched schedule.");
    return false;
  }

  runInfo.tiling_key = chequeIndex;
  runParams.sequenceLength = sequenceLength;
  runParams.dynamicrnnbatch = dynamicrnnbatch;
  runParams.chequeIndex = chequeIndex;
  // print tiling params
  PrintTilingParams(opType, runParams);
  SetRunningParams(runParams, runInfo);

  // block_dim, core num used in tik op
  // todo sync while dead
  runInfo.block_dim = DEFAULT_BLOCK_DIM;
  // workspace, null for tik op
  std::vector<int64_t> workspace = {4096};
  runInfo.workspaces = workspace;
  OP_LOGI(opType.c_str(), "tiling run success.");

  return true;
}

// register tiling interface of the DynamicRnn op
REGISTER_OP_TILING_FUNC_BUFFERED(DynamicRNN, DynamicRnnTiling);

}  // namespace optiling
