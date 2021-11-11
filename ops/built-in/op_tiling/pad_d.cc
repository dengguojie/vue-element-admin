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
 * \file pad_d.cpp
 * \brief
 */
#include "pad_common.h"
#include "../op_proto/util/error_util.h"
#include "error_log.h"

namespace optiling {
bool GetPadDCompileParams(const nlohmann::json& opCompileInfo, std::vector<std::vector<int64_t>>& padding, int& coreNum,
                          int& ubSize, int length) {
  using namespace nlohmann;
  auto allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("PadDTiling", "GetCompileParams, get core_num error");
    return false;
  }

  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("PadDTiling", "GetCompileParams, get ub_size error");
    return false;
  }

  coreNum = allVars["core_num"].get<std::int64_t>();
  ubSize = allVars["ub_size"].get<std::int64_t>();
  padding = allVars["padding"].get<std::vector<std::vector<int64_t>>>();
  if (int64_t(padding.size()) != length) {
    VECTOR_INNER_ERR_REPORT_TILIING("PadDTiling", "GetCompileParams, get padding error");
    return false;
  }

  return true;
}

bool PadDTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                OpRunInfo& runInfo) {
  using namespace ge;
  GELOGI("op[%s] PadDTiling running.", opType.c_str());

  // Get inShape, outShape
  padCommon pad;
  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [PadDTiling] : input shape error");
    return false;
  }

  if (opParas.outputs.empty() || opParas.outputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [PadDTiling] : output shape error");
    return false;
  }

  const std::vector<int64_t>& inShape = opParas.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& outShape = opParas.outputs[0].tensor[0].shape;
  const std::string dtype = opParas.inputs[0].tensor[0].dtype;
  bool ValidTensor = pad.CheckTensor(inShape, outShape);
  if (!ValidTensor) {
    return false;
  }

  // Get padding, ubSize, maxCore.
  int numBit = pad._numBit(dtype);
  int maxCore = 0;
  int ubSize = 0;
  std::vector<std::vector<int64_t>> padding;
  bool Success0 = GetPadDCompileParams(opCompileInfo, padding, maxCore, ubSize, int(inShape.size()));
  if (!Success0) {
    return false;
  }

  // Get n_inShape, n_outShape, n_padding after fused.
  std::vector<int64_t> n_inShape;
  std::vector<int64_t> n_outShape;
  std::vector<std::vector<int64_t>> n_padding;
  bool Success1 = pad.FusedAxis(n_inShape, n_outShape, n_padding, inShape, outShape, padding);
  if (!Success1) {
    return false;
  }

  /////////////////////////////////
  //---Get Params for Running----//
  /////////////////////////////////
  PadDTilingParams runParams;
  pad.InitTilingParams(runParams, int(n_inShape.size()));

  // Discriminate Align(1) and Not Align(0).
  runParams.branch = pad.CheckBranch(n_inShape, n_outShape, n_padding, numBit, 0) *
                     pad.CheckBranch(n_inShape, n_outShape, n_padding, numBit, 1);

  // Get Params In Circulation Layer
  pad.GetDepth(n_inShape, n_outShape, n_padding, runParams.depth, maxCore, numBit, runParams.branch);
  pad.GetCirculateParams("top", numBit, maxCore, n_inShape, n_outShape, n_padding, runParams);
  pad.GetCirculateParams("bottom", numBit, maxCore, n_inShape, n_outShape, n_padding, runParams);

  // Get Params In Recursion Layer
  if (runParams.branch == 1) {
    pad.GetRecurCore(runParams, n_inShape, n_outShape, n_padding, maxCore, numBit, ubSize);
  } else {
    pad.GetRecurCorePro(runParams, n_inShape, n_outShape, n_padding, maxCore, numBit, ubSize);
  }

  pad.SetRunningParams(runParams, runInfo);
  pad.PrintRunningParams(runParams);

  runInfo.block_dim = uint32_t(maxCore);
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;
  GELOGI("op[%s] tiling run success.", opType.c_str());
  return true;
}

// register tiling interface of the PadD op.
REGISTER_OP_TILING_FUNC_BUFFERED(PadD, PadDTiling);
}  // namespace optiling
