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
 * \file aipp.cc
 * \brief dynamic shape tiling of aipp
 */
#include <string>
#include <securec.h>
#include <algorithm>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "error_log.h"


namespace optiling {

struct AippTilingParams {
  int64_t needCoreNum;
  int64_t outputN;
  int64_t outputC1;
  int64_t outputH;
  int64_t outputW;
  int64_t outputC0;

  int64_t batchEachCore;
  int64_t batchLastCore;
};

void InitAippRunningParams(AippTilingParams& params) {
  params.needCoreNum = 0;
  params.outputN = 0;
  params.outputC1 = 0;
  params.outputH = 0;
  params.outputW = 0;
  params.outputC0 = 0;

  params.batchEachCore = 0;
  params.batchLastCore = 0;
}

void PrintAippTilingParams(const AippTilingParams& params) {
  OP_LOGD("Aipp", "op [AippTiling]  needCoreNum=%d.", params.needCoreNum);
  OP_LOGD("Aipp", "op [AippTiling]  outputN=%d.", params.outputN);
  OP_LOGD("Aipp", "op [AippTiling]  outputC1=%d.", params.outputC1);
  OP_LOGD("Aipp", "op [AippTiling]  outputH=%d.", params.outputH);
  OP_LOGD("Aipp", "op [AippTiling]  outputW=%d.", params.outputW);
  OP_LOGD("Aipp", "op [AippTiling]  outputC0=%d.", params.outputC0);
  OP_LOGD("Aipp", "op [AippTiling]  batchEachCore=%d.", params.batchEachCore);
  OP_LOGD("Aipp", "op [AippTiling]  batchLastCore=%d.", params.batchLastCore);
}

void SetAippRunningParams(const AippTilingParams& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.needCoreNum);
  ByteBufferPut(runInfo.tiling_data, params.outputN);
  ByteBufferPut(runInfo.tiling_data, params.outputC1);
  ByteBufferPut(runInfo.tiling_data, params.outputH);
  ByteBufferPut(runInfo.tiling_data, params.outputW);
  ByteBufferPut(runInfo.tiling_data, params.outputC0);

  ByteBufferPut(runInfo.tiling_data, params.batchEachCore);
  ByteBufferPut(runInfo.tiling_data, params.batchLastCore);
}

int64_t CalCeilValue(const std::string& opType, const int64_t& uValue, const int64_t& dValue) {
  int64_t resValue = 0;
  if (dValue == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [AippTiling]: CalCeilValue error, dValue is zero");
    return resValue;
  }

  resValue = (uValue + dValue - 1) / dValue;
  return resValue;
}

bool GetAippCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int64_t& coreNum) {
  using namespace nlohmann;
  auto allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [AippTiling]: GetAippCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();

  return true;
}

bool CalAippRunningParams(const std::string& opType, AippTilingParams& runParams, std::vector<int64_t> outputShape,
                          int64_t coreNum) {
  runParams.outputN = outputShape[0];
  runParams.outputC1 = outputShape[1];
  runParams.outputH = outputShape[2];
  runParams.outputW = outputShape[3];
  runParams.outputC0 = outputShape[4];
  OP_LOGD(opType.c_str(), "op [AippTiling], n=%ld, c1=%ld, h=%ld, w=%ld, c0=%ld", runParams.outputN,
          runParams.outputC1, runParams.outputH, runParams.outputW, runParams.outputC0);

  runParams.batchEachCore = CalCeilValue(opType, runParams.outputN, coreNum);
  runParams.needCoreNum = CalCeilValue(opType, runParams.outputN, runParams.batchEachCore);
  runParams.batchLastCore = runParams.outputN - (runParams.needCoreNum - 1) * runParams.batchEachCore;

  return true;
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] opCompileInfo: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool AippTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                OpRunInfo& runInfo) {
  using namespace ge;
  using namespace std;
  OP_LOGD(opType.c_str(), "op [AippTiling] running.");
  if (opCompileInfo == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op AippTiling: opCompileInfo json error.");
    return false;
  }
  if (opParas.inputs.size() < 2 || opParas.inputs[0].tensor.size() == 0 || opParas.inputs[1].tensor.size() == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op AippTiling: input shape error.");
    return false;
  }
  if (opParas.outputs.size() < 1 || opParas.outputs[0].tensor.size() == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op AippTiling: output shape error.");
    return false;
  }

  const std::vector<int64_t>& outputShape = opParas.outputs[0].tensor[0].shape;
  int64_t outputShapeSize = outputShape.size();
  if (outputShapeSize != 5 || outputShape[1] != 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op AippTiling: output shape dims must be 5, dim 1 must be equal to 1");
    return false;
  }
  std::string input_format = opParas.inputs[0].tensor[0].format;
  std::string output_format = opParas.outputs[0].tensor[0].format;
  OP_LOGD(opType.c_str(), "op [AippTiling], input format [%s], output format [%s]",
          input_format.c_str(), output_format.c_str());

  // get compile info
  int64_t coreNum = 0;
  if (!GetAippCompileParams(opType, opCompileInfo, coreNum) || coreNum == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op AippTiling: GetAippCompileParams error.");
    return false;
  }
  OP_LOGD(opType.c_str(), "op [AippTiling], coreNum is %ld.", coreNum);

  AippTilingParams runParams;
  InitAippRunningParams(runParams);
  if (!CalAippRunningParams(opType, runParams, outputShape, coreNum)) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op AippTiling: CalAippRunningParams failed.");
    return false;
  }
  SetAippRunningParams(runParams, runInfo);
  PrintAippTilingParams(runParams);

  runInfo.block_dim = runParams.needCoreNum;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  OP_LOGD(opType.c_str(), "op [AippTiling] run success.");

  return true;
}

// register tiling interface of the Aipp op
REGISTER_OP_TILING_FUNC_BUFFERED(Aipp, AippTiling);
}  // namespace optiling
