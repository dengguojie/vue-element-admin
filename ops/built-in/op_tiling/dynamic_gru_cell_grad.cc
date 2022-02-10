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

#include <iostream>
#include "vector_tiling.h"
#include "error_log.h"
#include "op_log.h"

namespace {
  constexpr int32_t NUM_SIXTEEN = 16;
  constexpr int32_t NUM_EIGHT = 8;
  constexpr int32_t NUM_TWO = 2;
  constexpr int32_t ALLIGN_SIZE = 256;
}

namespace optiling {
static const int INPUT_SIZE = 9;
static std::map<std::string, int64_t> TYPE_SIZE = {{"float16", 2}, {"float32", 4}};

bool GetCompileParams(const nlohmann::json& opCompileInfo, int64_t& coreNum, int64_t& ubByteSize) {
  using namespace nlohmann;
  auto allVars = opCompileInfo["vars"];
  if (allVars.count("device_aicore_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("DynamicGRUCellGradTiling", "GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["device_aicore_num"].get<std::int64_t>();

  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("DynamicGRUCellGradTiling", "GetCompileParams, get ub_size error");
    return false;
  }
  ubByteSize = allVars["ub_size"].get<std::int64_t>();

  OP_LOGD("op [DynamicGRUCellGrad] : get ub_size %d.", ubByteSize);
  OP_LOGD("op [DynamicGRUCellGrad] : get coreNum %d.", coreNum);
  return true;
}

bool CheckGRUGradInput(const TeOpParas& op_paras) {
  if (op_paras.inputs.size() < INPUT_SIZE) {
    VECTOR_INNER_ERR_REPORT_TILIING("DynamicGRUCellGradTiling", "opParas.inputs.size error");
    return false;
  }

  for (int i = 0; i < INPUT_SIZE; i++) {
    if (op_paras.inputs[i].tensor.size() == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("DynamicGRUCellGradTiling", "opParas.input[%d].tensor.size error", i);
      return false;
    }
  }
  return true;
}

void SetGRUGradRunInfo(map<std::string, int64_t> tilingPara, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, tilingPara["coreNum"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["loopNum"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["loopEle"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["blockSize"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["tailNum"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["tailCoreNum"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["tailLoopEle"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["tailLastEle"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["tSize"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["fuseSize"]);
}

void PrintGRUGradParams(const std::string& op_type, map<std::string, int64_t> tilingPara) {
  OP_LOGD(op_type.c_str(), "DynamicGRUCellGradTiling: coreNum=%lld, loopNum=%lld, \
          loopEle=%lld, blockSize=%lld, tailNum=%lld, tailCoreNum=%lld, tailLoopEle=%lld,\
	      tailLastEle=%lld, tSize=%lld, fuseSize=%lld.", tilingPara["coreNum"], tilingPara["loopNum"],
          tilingPara["loopEle"], tilingPara["blockSize"], tilingPara["tailNum"], tilingPara["tailCoreNum"],
          tilingPara["tailLoopEle"], tilingPara["tailLastEle"], tilingPara["tSize"], tilingPara["fuseSize"]);
}

bool DynamicGRUCellGradTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                              OpRunInfo& run_info) {
  using namespace ge;

  OP_LOGD("op [%s] Enter DynamicGRUCellGradTiling inputs size:%d", op_type.c_str(), op_paras.inputs.size());
  if (op_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op DynamicGRUCellGradTiling: op_info json error.");
    return false;
  }

  std::int64_t deviceCoreNum = 0;
  std::int64_t ubByteSize = 0;

  bool getInfoStatus = GetCompileParams(op_info, deviceCoreNum, ubByteSize);
  if (!getInfoStatus || deviceCoreNum == 0 || ubByteSize == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Tiling: DynamicGRUCellGradTiling error.");
    return false;
  }

  // check input
  if (!CheckGRUGradInput(op_paras)) {
    return false;
  }

  // get h data dyte
  std::string hDType = op_paras.inputs[0].tensor[0].dtype;
  // get input data type size
  int64_t dataTypeSize = TYPE_SIZE[hDType];
  int64_t t_state_and_tiling_size = (NUM_SIXTEEN * dataTypeSize + NUM_SIXTEEN * NUM_EIGHT) * NUM_TWO;
  int64_t ub_max_ele_num = (ubByteSize - t_state_and_tiling_size) / dataTypeSize;
  int64_t align = ALLIGN_SIZE / dataTypeSize;

  // get tiling para
  std::map<std::string, int64_t> tilingPara = {{"coreNum", 0}, {"loopNum", 0},     {"loopEle", 0},     {"blockSize", 0},
                                          {"tailNum", 0}, {"tailCoreNum", 0}, {"tailLoopEle", 0}, {"tailLastEle", 0},
                                          {"tSize", 0},   {"fuseSize", 0}};
  const std::vector<int64_t>& dnxShape = op_paras.outputs[2].tensor[0].shape;
  int64_t shapeSize = dnxShape[0];
  for (size_t i = 1; i < dnxShape.size(); i++) {
    shapeSize = shapeSize * dnxShape[i];
  }

  // set tSize
  tilingPara["tSize"] = op_paras.inputs[2].tensor[0].shape[0];
  // set fuseSize
  tilingPara["fuseSize"] = shapeSize;
  if (shapeSize < align * deviceCoreNum) {
    tilingPara["coreNum"] = shapeSize / align;
    tilingPara["loopNum"] = 1;
    tilingPara["loopEle"] = align;
    tilingPara["blockSize"] = align;
    tilingPara["tailNum"] = 0;
    tilingPara["tailCoreNum"] = 0;
    tilingPara["tailLoopEle"] = 0;
    tilingPara["tailLastEle"] = 0;
    run_info.block_dim = tilingPara["coreNum"];
    SetGRUGradRunInfo(tilingPara, run_info);
    PrintGRUGradParams(op_type, tilingPara);
    return true;
  }
  int64_t coreNum = deviceCoreNum;
  int64_t maxBlockEleNum = (ub_max_ele_num / NUM_EIGHT / NUM_TWO / align) * align;
  int64_t loopNum = shapeSize / (maxBlockEleNum * coreNum);
  int64_t loopEle = maxBlockEleNum;
  int64_t blockSize = loopEle * loopNum;
  int64_t tailNum = shapeSize - blockSize * coreNum;
  tilingPara["coreNum"] = coreNum;
  tilingPara["loopNum"] = loopNum;
  tilingPara["loopEle"] = loopEle;
  tilingPara["blockSize"] = blockSize;
  tilingPara["tailNum"] = tailNum;
  if (tailNum == 0) {
    tilingPara["tailCoreNum"] = 0;
    tilingPara["tailLoopEle"] = 0;
    tilingPara["tailLastEle"] = 0;
  } else {
    int64_t tailLoopEle = (tailNum / coreNum + align - 1) / align * align;
    tilingPara["tailLoopEle"] = tailLoopEle;
    tilingPara["tailCoreNum"] = (tailNum + tailLoopEle - 1) / tailLoopEle;
    tilingPara["tailLastEle"] = (tailNum % tailLoopEle == 0) ? tailLoopEle : tailNum % tailLoopEle;
  }

  PrintGRUGradParams(op_type, tilingPara);
  run_info.block_dim = tilingPara["coreNum"];
  SetGRUGradRunInfo(tilingPara, run_info);
  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;
  OP_LOGI("op[%s] tiling run success.", op_type.c_str());
  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(DynamicGRUCellGrad, DynamicGRUCellGradTiling);
}  // namespace optiling
// namespace optiling
