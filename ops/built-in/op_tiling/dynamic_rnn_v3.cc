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
 * \file dynamic_rnn_v3.cpp
 * \brief tiling function of op
 */
#include <string>
#include <nlohmann/json.hpp>
#include "op_tiling.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"

namespace optiling {
const int DEFAULT_SHAPE_LIST_SIZE = 3;
const int DEFAULT_INDEX_TWO = 2;
const int DEFAULT_RETURN = -2;
const int DEFAULT_PARAS_INPUT_SIZE = 3;
const int DEFAULT_XSHAPE_SIZE = 3;
const int DEFAULT_BLOCK_DIM = 32;
const int WORKSPACE_SIZE = 4096;
const int NUM_SIXTEEN = 16;
const int NUM_FIFTEEN = 15;

struct DynamicRnnV3Param {
  int32_t sequenceLength{0};
  int32_t dynamicRnnBatch{0};
  int32_t chequeIndex{-1};
};

void SetRunningParams(const DynamicRnnV3Param& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.sequenceLength);
  ByteBufferPut(runInfo.tiling_data, runParams.dynamicRnnBatch);
  ByteBufferPut(runInfo.tiling_data, runParams.chequeIndex);
}

/*
 * @brief: print function of op
 * @param [in] opType: opType of the op
 * @param [in] params: tilling params
 * @return void: void
 */
void PrintTilingParams(const std::string& opType, const DynamicRnnV3Param& params) {
  OP_LOGD(opType.c_str(), "sequenceLength=%lld", params.sequenceLength);
  OP_LOGD(opType.c_str(), "dynamicRnnBatch=%lld", params.dynamicRnnBatch);
  OP_LOGD(opType.c_str(), "chequeIndex=%lld", params.chequeIndex);
}

// return tiling_indextiling_index
int32_t GetRnnV3LibItem(const std::string &opType, const nlohmann::json &opCompileInfoJson,
                        std::vector<int64_t> xShape) {
  OP_LOGD(opType.c_str(), "enter DynamicRnnV3Tiling GetRnnV3LibItem");
  const nlohmann::json& allVars = opCompileInfoJson["vars"];
  if (allVars.empty()) {
    OP_LOGE(opType.c_str(), "DynamicRnnV3Tiling: GetRnnV3LibItem, get vars failed.");
    return DEFAULT_RETURN;
  }
  std::vector<std::vector<int64_t>> tune_shape_list;
  tune_shape_list = allVars.at("tune_shape_list").get<std::vector<std::vector<int64_t>>>();
  if (tune_shape_list.empty()) {
    OP_LOGE(opType.c_str(), "DynamicRnnV3Tiling: GetRnnV3LibItem, get tune_shape_list failed.");
    return DEFAULT_RETURN;
  }

  for (uint64_t i = 0; i < tune_shape_list.size(); i++) {
    if (tune_shape_list[i].size() < DEFAULT_SHAPE_LIST_SIZE) {
      OP_LOGE(opType.c_str(), "tune_shape_list's size is illegal. it's %lu", tune_shape_list[i].size());
      return DEFAULT_RETURN;
    }
    if ((tune_shape_list[i][0] == -1) && (tune_shape_list[i][1] == -1)) {
      OP_LOGD(opType.c_str(), "matched floor schedule, the matched schedule is %lld",
              tune_shape_list[i][DEFAULT_INDEX_TWO]);
      int32_t res = static_cast<int32_t>(tune_shape_list[i][DEFAULT_INDEX_TWO]);
      return res;
    }
    if ((tune_shape_list[i][0] == xShape[0]) &&
        (((tune_shape_list[i][1] + NUM_FIFTEEN) / NUM_SIXTEEN) == xShape[DEFAULT_INDEX_TWO])) {
      OP_LOGD(opType.c_str(), "The matched schedule is %lld", tune_shape_list[i][DEFAULT_INDEX_TWO]);
      int32_t res = static_cast<int32_t>(tune_shape_list[i][DEFAULT_INDEX_TWO]);
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
bool DynamicRnnV3Tiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                        OpRunInfo& runInfo) {
  OP_LOGI(opType.c_str(), "tiling running.");
  if (op_info == nullptr) {
    OP_LOGE(opType.c_str(), "op DynamicRnnV3Tiling: op_info json error.");
    return false;
  }
  if (opParas.inputs.empty() || opParas.inputs.size() < DEFAULT_PARAS_INPUT_SIZE || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty() || opParas.inputs[2].tensor.empty()) {
    OP_LOGE(opType.c_str(), "op DynamicRnnV3Tiling: input shape error.");
    return false;
  }
  std::vector<int64_t> xShape = opParas.inputs[0].tensor[0].shape;
  if (xShape.size() < DEFAULT_XSHAPE_SIZE) {
    OP_LOGE(opType.c_str(), "op DynamicRnnV3Tiling: inputs shape are invalid.");
    return false;
  }

  // init running parameters
  DynamicRnnV3Param runParams;

  // set run tiling data
  int32_t sequenceLength = xShape[0];
  int32_t dynamicRnnBatch = xShape[DEFAULT_INDEX_TWO];
  int32_t chequeIndex = GetRnnV3LibItem(opType, op_info, xShape);
  if (chequeIndex == DEFAULT_RETURN) {
    OP_LOGE(opType.c_str(), "DynamicRnnV3Tiling has no matched schedule.");
    return false;
  }

  runInfo.tiling_key = chequeIndex;
  runParams.sequenceLength = sequenceLength;
  runParams.dynamicRnnBatch = dynamicRnnBatch;
  runParams.chequeIndex = chequeIndex;
  // print tiling params
  PrintTilingParams(opType, runParams);
  SetRunningParams(runParams, runInfo);

  // block_dim, core num used in tik op
  // todo sync while dead
  runInfo.block_dim = DEFAULT_BLOCK_DIM;
  // workspace, null for tik op
  std::vector<int64_t> workspace = {WORKSPACE_SIZE};
  runInfo.workspaces = workspace;
  OP_LOGI(opType.c_str(), "DynamicRnnV3Tiling run success.");

  return true;
}

// register tiling interface of the DynamicRnnV3 op
REGISTER_OP_TILING_FUNC_BUFFERED(DynamicRNNV3, DynamicRnnV3Tiling);
}  // namespace optiling
