/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file dynamic_gru_v2.cpp
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

namespace {
  constexpr int32_t INPUT_BASE_SIZE = 3;
}

namespace optiling {
static const int WORKSPACE_SIZE = 4096;
static const int BLOCK_SIZE = 4096;

struct DynamicGruV2Param {
  int32_t sequenceLength;
  int32_t dynamicgruBatch;
  int32_t chequeIndex;
};

bool CheckInputShape(const std::string& opType, std::vector<int64_t> xShape) {
  int32_t xDims = xShape.size();
  OP_LOGD(opType.c_str(), "op [DynamicGruV2Tiling] : xDims=%d", xDims);

  if (xShape[0] <= 0) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices", "the first dim of x must be greater than 0");
    OP_LOGE(opType.c_str(), "op [DynamicGruV2Tiling] : CheckInputShape, x.shape is invalid.");
    return false;
  }

  return true;
}

void InitRunningParams(DynamicGruV2Param& params) {
  params.sequenceLength = 0;
  params.dynamicgruBatch = 0;
  params.chequeIndex = -1;
}

void SetRunningParams(const DynamicGruV2Param& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.sequenceLength);
  ByteBufferPut(runInfo.tiling_data, runParams.dynamicgruBatch);
  ByteBufferPut(runInfo.tiling_data, runParams.chequeIndex);
}

/*
 * @brief: print function of op
 * @param [in] opType: opType of the op
 * @param [in] params: tilling params
 * @return void: void
 */
void PrintTilingParams(const std::string& op_type, const DynamicGruV2Param& params) {
  OP_LOGD(op_type.c_str(), "params.sequenceLength=%d", params.sequenceLength);
  OP_LOGD(op_type.c_str(), "params.dynamicgruBatch=%d", params.dynamicgruBatch);
  OP_LOGD(op_type.c_str(), "params.chequeIndex=%d", params.chequeIndex);
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool DynamicGruV2Tiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                        OpRunInfo& runInfo) {
  OP_LOGI(opType.c_str(), "op tiling running.");
  if (op_info == nullptr) {
    OP_LOGE(opType.c_str(), "op DynamicGruV2Tiling: op_info json error.");
    return false;
  }

  if (opParas.inputs.size() < INPUT_BASE_SIZE || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty()) {
    OP_LOGE(opType.c_str(), "op DynamicGruV2Tiling: input shape error.");
    return false;
  }

  std::vector<int64_t> xShape = opParas.inputs[0].tensor[0].shape;

   // check inputs shape
  bool ret = CheckInputShape(opType, xShape);
  if (!ret) {
    OP_LOGE(opType.c_str(), "op DynamicGruV2Tiling: inputs shape are invalid.");
    return ret;
  }

  // init running parameters
  DynamicGruV2Param runParams;
  InitRunningParams(runParams);

  // set run tiling data
  int32_t sequenceLength = xShape[0];
  int32_t dynamicgruBatch = xShape[2];
  // default index
  int32_t chequeIndex = 0;
  runInfo.tiling_key = chequeIndex;
  OP_LOGD(opType.c_str(), "sequenceLength=%d.", sequenceLength);
  OP_LOGD(opType.c_str(), "dynamicgruBatch=%d.", dynamicgruBatch);
  OP_LOGD(opType.c_str(), "chequeIndex=%d.", chequeIndex);
  runParams.sequenceLength = sequenceLength;
  runParams.dynamicgruBatch = dynamicgruBatch;
  runParams.chequeIndex = chequeIndex;
  SetRunningParams(runParams, runInfo);

  // print tiling params
  PrintTilingParams(opType, runParams);

  // block_dim, core num used in tik op
  // todo sync while dead
  runInfo.block_dim = BLOCK_SIZE;
  // workspace, null for tik op
  std::vector<int64_t> workspace={WORKSPACE_SIZE};
  runInfo.workspaces = workspace;

  OP_LOGI(opType.c_str(), "tiling run success.");

  return true;
}

// register tiling interface of the DynamicGruV2 op
REGISTER_OP_TILING_FUNC_BUFFERED(DynamicGRUV2, DynamicGruV2Tiling);
}  // namespace optiling
