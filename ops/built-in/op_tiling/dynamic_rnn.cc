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

struct DynamicRnnParam {
  int32_t sequenceLength;
  int32_t dynamicrnnbatch;
  int32_t cheque_index;
};

bool GetCompileParams(const std::string& opType,
                      const nlohmann::json& opCompileInfoJson) {
  using namespace nlohmann;

  const auto& allVars = opCompileInfoJson["vars"];

  if (allVars.count("tune_shape_list") != 0) {
    OP_LOGE(opType.c_str(), "op [DynamicRnnTiling] : GetCompileParams, get tune_shape_list success.");
  } else {
    OP_LOGE(opType.c_str(), "op [DynamicRnnTiling] : GetCompileParams, get tune_shape_list error.");
    return false;
  }

  return true;
}

bool CheckTensorShape(const std::string& opType, std::vector<int64_t> xShape) {
  int32_t xDims = xShape.size();
  GELOGD("op [DynamicRnnTiling] : xDims=%d", xDims);

  if (xShape[0] <= 0) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices", "the first dim of x must be greater than 0");
    OP_LOGE(opType.c_str(), "op [DynamicRnnTiling] : CheckTensorShape, x.shape is invalid.");
    return false;
  }

  return true;
}

void InitRunningParams(DynamicRnnParam& params) {
  params.sequenceLength = 0;
  params.dynamicrnnbatch = 0;
  params.cheque_index = -1;
}

void SetRunningParams(const DynamicRnnParam& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.sequenceLength);
  ByteBufferPut(runInfo.tiling_data, runParams.dynamicrnnbatch);
  ByteBufferPut(runInfo.tiling_data, runParams.cheque_index);
}

/*
 * @brief: print function of op
 * @param [in] opType: opType of the op
 * @param [in] params: tilling params
 * @return void: void
 */
void PrintTilingParams(const std::string& op_type, const DynamicRnnParam& params) {
  GELOGD("op [%s] : params.sequenceLength=%d", op_type.c_str(), params.sequenceLength);
  GELOGD("op [%s] : params.dynamicrnnbatch=%d", op_type.c_str(), params.dynamicrnnbatch);
  GELOGD("op [%s] : params.cheque_index=%d", op_type.c_str(), params.cheque_index);
}

// return tiling_indextiling_index
int32_t GetLibItem(const std::string& opType, const nlohmann::json& opCompileInfoJson, std::vector<int64_t> xShape) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfoJson["vars"];

  if (allVars.count("tune_shape_list") == 0) {
    OP_LOGE(opType.c_str(), "op [DynamicRnnTiling] : GetLibItem, get tune_shape_list failed-------");
    return -1;
  }
  try {
  std::vector<std::vector<int64_t>> tune_shape_list = allVars["tune_shape_list"].get<std::vector<std::vector<int64_t>>>();
  std::vector<int64_t> tune_shape_list_value = tune_shape_list[0];
    for (int64_t i = 0; i < tune_shape_list.size(); i++) {
      if ((tune_shape_list[i][0] == xShape[0]) && ((tune_shape_list[i][1] / 16) == xShape[2])) {
        OP_LOGE(opType.c_str(), "op [DynamicRnnTiling] : GetLibItem, The corresponding schedule is", tune_shape_list[i][2]);
        int32_t res = (int32_t)tune_shape_list[i][2];
        return res;
      }
    }
  } catch (const std::exception &e) {
    OP_LOGE(opType.c_str(), "Failed to get lib item error. Error message: %s", e.what());
  }
  OP_LOGE(opType.c_str(), "op [DynamicRnnTiling] : GetLibItem, No matching schedule is found.-------");
  return -1;
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
  GELOGI("op[%s] tiling running.", opType.c_str());
  if (op_info == nullptr) {
    OP_LOGE(opType.c_str(), "op DynamicRnnTiling: op_info json error.");
    return false;
  }

  if (opParas.inputs.empty() || opParas.inputs.size() < 2 || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "x or indices",
                                  "The length of inputs is less than 2 or the inputs is empty");
    OP_LOGE(opType.c_str(), "op DynamicRnnTiling: input shape error.");
    return false;
  }

  std::vector<int64_t> xShape = opParas.inputs[0].tensor[0].shape;

   // check inputs shape
  bool ret = CheckTensorShape(opType, xShape);
  if (!ret) {
    OP_LOGE(opType.c_str(), "op DynamicRnnTiling: inputs shape are invalid.");
    return ret;
  }

  bool flag = GetCompileParams(opType, op_info);
  if (!flag) {
    OP_LOGE(opType.c_str(), "op DynamicRnnTiling: GetCompileParams error.");
    return false;
  }

  // init running parameters
  DynamicRnnParam runParams;
  InitRunningParams(runParams);

  // set run tiling data
  int32_t sequenceLength = xShape[0];
  int32_t dynamicrnnbatch = xShape[2];
  int32_t cheque_index = GetLibItem(opType, op_info, xShape);
  runInfo.tiling_key = cheque_index;
  GELOGD("op [DynamicRnnTiling] : sequenceLength=%d.", sequenceLength);
  GELOGD("op [DynamicRnnTiling] : dynamicrnnbatch=%d.", dynamicrnnbatch);
  GELOGD("op [DynamicRnnTiling] : cheque_index=%d.", cheque_index);
  runParams.sequenceLength = sequenceLength;
  runParams.dynamicrnnbatch = dynamicrnnbatch;
  runParams.cheque_index = cheque_index;
  SetRunningParams(runParams, runInfo);

  // print tiling params
  PrintTilingParams(opType, runParams);

  // block_dim, core num used in tik op
  // todo sync while dead
  runInfo.block_dim = 32;
  // workspace, null for tik op
  std::vector<int64_t> workspace={4096};
  runInfo.workspaces = workspace;

  GELOGI("op[%s] tiling run success.", opType.c_str());

  return true;
}

// register tiling interface of the DynamicRnn op
REGISTER_OP_TILING_FUNC_BUFFERED(DynamicRNN, DynamicRnnTiling);

}  // namespace optiling
