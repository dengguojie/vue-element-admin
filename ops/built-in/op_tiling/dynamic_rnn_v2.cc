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
 * \file dynamic_rnn_v2.cpp
 * \brief tiling function of op
 */
#include <string>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"

namespace {
  constexpr int32_t DST_SHAPE_SIZE = 3;
  constexpr int32_t DST_INPUT_SIZE = 2;
  constexpr int32_t NUM_SIXTEEN = 16;
  constexpr int32_t INPUT_INDEX_TWO = 2;
}

namespace optiling {
static const int32_t MAX_BLOCK_DIM = 32;

struct DynamicRnnV2Param {
  int32_t sequenceLength{0};
  int32_t dynamicRnnBatch{0};
  int32_t chequeIndex{-1};
};

void SetRnnV2RunningParams(const DynamicRnnV2Param& runParams, OpRunInfo& runInfo) {
  OP_LOGD("DynamicRnnV2", "op [DynamicRnnV2Tiling] enter SetRnnV2RunningParams");
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
void PrintRnnV2TilingParams(const std::string& op_type, const DynamicRnnV2Param& params) {
  OP_LOGD(op_type.c_str(), "op [DynamicRnnV2Tiling] params.sequenceLength=%d", params.sequenceLength);
  OP_LOGD(op_type.c_str(), "op [DynamicRnnV2Tiling] params.dynamicRnnBatch=%d", params.dynamicRnnBatch);
  OP_LOGD(op_type.c_str(), "op [DynamicRnnV2Tiling] params.chequeIndex=%d", params.chequeIndex);
}

// return tiling_index
int32_t GetRnnV2LibItem(const std::string& opType, const nlohmann::json& opCompileInfoJson,
                        std::vector<int64_t> xShape) {
  OP_LOGD("DynamicRnnV2", "op [DynamicRnnV2Tiling] enter GetRnnV2LibItem");
  const nlohmann::json& allVars = opCompileInfoJson["vars"];
  if (allVars.empty()) {
    OP_LOGE(opType.c_str(), "op [DynamicRnnV2Tiling] : GetRnnV2LibItem, get vars failed.");
    return -1;
  }
  std::vector<std::vector<int64_t>> tune_shape_list;
  tune_shape_list = allVars["tune_shape_list"].get<std::vector<std::vector<int64_t>>>();
  if (tune_shape_list.empty()) {
    OP_LOGE(opType.c_str(), "op [DynamicRnnV2Tiling] : GetRnnV2LibItem, get tune_shape_list failed.");
    return -1;
  }
  for (uint64_t i = 0; i < tune_shape_list.size(); i++) {
    if (tune_shape_list[i].size() < DST_SHAPE_SIZE) {
      OP_LOGE(opType.c_str(), "op [DynamicRnnV2Tiling] : GetRnnV2LibItem, No matching schedule is found.");
      return -1;
    }
    if ((tune_shape_list[i][0] == -1) && (tune_shape_list[i][1] == -1)) {
      OP_LOGI(opType.c_str(), "op [DynamicRnnV2Tiling] : GetRnnV2LibItem, The corresponding schedule is",
              tune_shape_list[i][2]);
      return (int32_t)tune_shape_list[i][2];
    }
    if ((tune_shape_list[i][0] == xShape[0]) && ((tune_shape_list[i][1] / NUM_SIXTEEN) == xShape[INPUT_INDEX_TWO])) {
      OP_LOGI(opType.c_str(), "op [DynamicRnnV2Tiling] : GetRnnV2LibItem, The corresponding schedule is",
              tune_shape_list[i][2]);
      return (int32_t)tune_shape_list[i][2];
    }
  }
  OP_LOGE(opType.c_str(), "op [DynamicRnnV2Tiling] : GetRnnV2LibItem, No matching schedule is found.");
  return -1;
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] opInfo: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool DynamicRnnV2Tiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opInfo,
                        OpRunInfo& runInfo) {
  OP_LOGI(opType.c_str(), "DynamicRnnV2Tiling running.");
  if (opInfo == nullptr) {
    OP_LOGE(opType.c_str(), "op DynamicRnnV2Tiling: opInfo json error.");
    return false;
  }

  if (opParas.inputs.empty() || opParas.inputs.size() < DST_INPUT_SIZE || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "x or indices",
                                  "The length of inputs is less than 2 or the inputs is empty");
    OP_LOGE(opType.c_str(), "op DynamicRnnV2Tiling: input shape error.");
    return false;
  }

  std::vector<int64_t> xShape = opParas.inputs[0].tensor[0].shape;
  if (xShape.size() < DST_SHAPE_SIZE) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "x", "x.shape is invalid");
    OP_LOGE(opType.c_str(), "op [DynamicRnnV2Tiling] : CheckRnnV2TensorShape, x.shape is invalid.");
    return false;
  }

  // init running parameters
  DynamicRnnV2Param runParams;
  int32_t sequenceLength = xShape[0];
  int32_t dynamicRnnBatch = xShape[2];
  int32_t chequeIndex = GetRnnV2LibItem(opType, opInfo, xShape);
  if (chequeIndex == -1) {
    OP_LOGE(opType.c_str(), "op [DynamicRnnV2Tiling] : No matching schedule is found.");
    return false;
  }
  runInfo.tiling_key = chequeIndex;
  runParams.sequenceLength = sequenceLength;
  runParams.dynamicRnnBatch = dynamicRnnBatch;
  runParams.chequeIndex = chequeIndex;
  SetRnnV2RunningParams(runParams, runInfo);

  // print tiling params
  PrintRnnV2TilingParams(opType, runParams);

  // block_dim, core num used in tik op
  runInfo.block_dim = MAX_BLOCK_DIM;
  // workspace, null for tik op
  std::vector<int64_t> workspace = {512 * 8, 1024 * 1024 * 4, 1024 * 1024 * 4};
  runInfo.workspaces = workspace;

  OP_LOGI(opType.c_str(), "DynamicRnnV2Tiling end.");
  return true;
}

// register tiling interface of the DynamicRnn op
REGISTER_OP_TILING_FUNC_BUFFERED(DynamicRNNV2, DynamicRnnV2Tiling);
}  // namespace optiling
