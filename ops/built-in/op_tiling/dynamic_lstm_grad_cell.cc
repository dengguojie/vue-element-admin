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

using namespace std;

namespace optiling {
static const int INPUT_SIZE = 11;
const int64_t BLOCK_SIZE = 16;
const int64_t TOTAL_INPUT_NUMS_NO_MASK = 20;
static map<std::string, int64_t> TYPE_SIZE = {
    {
        "float16", 2
    }, {
        "float32", 4
    }
};


bool GetCompileParams(const nlohmann::json& opCompileInfo, int64_t& coreNum, int64_t& ubByteSize, bool& maskInput) {
  using namespace nlohmann;
  auto allVars = opCompileInfo["vars"];
  if (allVars.count("device_aicore_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("DynamicLSTMGradCellTiling", "GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["device_aicore_num"].get<std::int64_t>();

  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("DynamicLSTMGradCellTiling", "GetCompileParams, get ub_size error");
    return false;
  }
  ubByteSize = allVars["ub_size"].get<std::int64_t>();
  if (allVars.count("mask_input") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("DynamicLSTMGradCellTiling", "GetCompileParams, get mask_input error");
    return false;
  }
  if (allVars["mask_input"].get<std::int64_t>() == 0) {
    maskInput = false;
  }
  GELOGD("op [DynamicLSTMGradCellTiling] : get ub_size %d.", ubByteSize);
  GELOGD("op [DynamicLSTMGradCellTiling] : get coreNum %d.", coreNum);
  GELOGD("op [DynamicLSTMGradCellTiling] : get maskInput %d.", maskInput);
  return true;
}

bool CheckInput(const TeOpParas& op_paras) {
  if (op_paras.inputs.size() < INPUT_SIZE) {
    VECTOR_INNER_ERR_REPORT_TILIING("DynamicLSTMGradCellTiling", "opParas.inputs.size error");
    return false;
  }

  for (int i = 0; i < INPUT_SIZE; i++) {
    if (op_paras.inputs[i].tensor.size() == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("DynamicLSTMGradCellTiling",  "opParas.input[%d].tensor.size error", i);
      return false;
    }
  }
  return true;
}

void setRunInfo(map<std::string, int64_t> tilingPara, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, tilingPara["tSize"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["eleEachCore"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["outLoopNum"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["outLoopEleNum"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["innerLoopNum"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["innerLoopEleNum"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["lastLoopEleNum"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["ubSize"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["hiddenSize"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["batchSize"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["useCoreNum"]);
  ByteBufferPut(runInfo.tiling_data, tilingPara["fuseSize"]);
}

void PrintParams(map<std::string, int64_t> tilingPara) {
  GELOGD("op [DynamicLSTMGradCellTilling] : tSize=%d.", tilingPara["tSize"]);
  GELOGD("op [DynamicLSTMGradCellTilling] : eleEachCore=%d.", tilingPara["eleEachCore"]);
  GELOGD("op [DynamicLSTMGradCellTilling] : outLoopNum=%d.", tilingPara["outLoopNum"]);
  GELOGD("op [DynamicLSTMGradCellTilling] : outLoopEleNum=%d.", tilingPara["outLoopEleNum"]);
  GELOGD("op [DynamicLSTMGradCellTilling] : innerLoopNum=%d.", tilingPara["innerLoopNum"]);
  GELOGD("op [DynamicLSTMGradCellTilling] : innerLoopEleNum=%d.", tilingPara["innerLoopEleNum"]);
  GELOGD("op [DynamicLSTMGradCellTilling] : lastLoopEleNum=%d.", tilingPara["lastLoopEleNum"]);
  GELOGD("op [DynamicLSTMGradCellTilling] : hiddenSize=%d.", tilingPara["hiddenSize"]);
  GELOGD("op [DynamicLSTMGradCellTilling] : batchSize=%d.", tilingPara["batchSize"]);
  GELOGD("op [DynamicLSTMGradCellTilling] : useCoreNum=%d.", tilingPara["useCoreNum"]);
  GELOGD("op [DynamicLSTMGradCellTilling] : fuseSize=%d.", tilingPara["fuseSize"]);
  GELOGD("op [DynamicLSTMGradCellTilling] : ubSize=%d.", tilingPara["ubSize"]);
}

bool DynamicLSTMGradCellTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                               OpRunInfo& run_info) {
  using namespace ge;
  using namespace std;

  GELOGD("op [%s] Enter DynamicLSTMGradCellTilling inputs size:%d", op_type.c_str(), op_paras.inputs.size());
  if (op_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op DynamicLSTMGradCellTilling: op_info json error.");
    return false;
  }

  int64_t deviceCoreNum = 0;
  int64_t ubByteSize = 0;
  bool maskInput = true;
  bool getInfoStatus = GetCompileParams(op_info, deviceCoreNum, ubByteSize, maskInput);
  GELOGD("op [%s] Enter DynamicLSTMGradCellTilling check input pass0", op_type.c_str());
  if (!getInfoStatus || deviceCoreNum == 0 || ubByteSize == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Tiling: DynamicLSTMGradCellTilling error.");
    return false;
  }
  GELOGD("op [%s] Enter DynamicLSTMGradCellTilling check input pass1", op_type.c_str());

  // check input
  if (!CheckInput(op_paras)) {
    return false;
  }
  GELOGD("op [%s] Enter DynamicLSTMGradCellTilling check input pass2", op_type.c_str());
  GELOGD("op [DynamicLSTMGradCellTilling] : deviceCoreNum=%d.", deviceCoreNum);
  GELOGD("op [DynamicLSTMGradCellTilling] : ubByteSize=%d.", ubByteSize);
  const std::vector<int64_t>& dhShape = op_paras.inputs[3].tensor[0].shape;
  GELOGD("op [%s] Enter DynamicLSTMGradCellTilling check input pass2.2", op_type.c_str());
  int64_t shapeSize = dhShape[0];
  GELOGD("op [%s] Enter DynamicLSTMGradCellTilling check input pass2.1", op_type.c_str());
  GELOGD("op [DynamicLSTMGradCellTilling] : i %d=%d.", 0, dhShape[0]);
  for (size_t i = 1; i < dhShape.size(); i++) {
    GELOGD("op [DynamicLSTMGradCellTilling] : i %d=%d.", i, dhShape[i]);
    shapeSize *= dhShape[i];
  }
  GELOGD("op [%s] Enter DynamicLSTMGradCellTilling check input pass3", op_type.c_str());
  GELOGD("op [DynamicLSTMGradCellTilling] : shapeSize=%d.", shapeSize);

  // init tiling para
  map<std::string, int64_t> tilingPara = {
      {"tSize", 0 }, {"eleEachCore", 0}, {"outLoopNum", 0}, {"outLoopEleNum", 0}, {"innerLoopNum", 0},
      {"innerLoopEleNum", 0}, {"lastLoopEleNum", 0}, {"ubSize", 0}, {"hiddenSize", 0}, {"batchSize", 0},
      {"useCoreNum", 0}, {"fuseSize", 0},
  };

  int64_t hiddenSize = op_paras.inputs[2].tensor[0].shape[1];
  int64_t batchSize = op_paras.inputs[2].tensor[0].shape[2];
  tilingPara["hiddenSize"] = hiddenSize * BLOCK_SIZE;
  tilingPara["batchSize"] = batchSize * BLOCK_SIZE;
  if (op_paras.inputs[2].tensor[0].format != "FRACTAL_NZ") {
    hiddenSize = op_paras.inputs[2].tensor[0].shape[2];
    batchSize = op_paras.inputs[2].tensor[0].shape[1];
    tilingPara["hiddenSize"] = hiddenSize;
    tilingPara["batchSize"] = batchSize;
  }

  tilingPara["useCoreNum"] = hiddenSize * batchSize >= deviceCoreNum ? deviceCoreNum : hiddenSize * batchSize;
  tilingPara["fuseSize"] = shapeSize;
  std::string cDType = op_paras.inputs[1].tensor[0].dtype;
  int64_t dataTypeSize = TYPE_SIZE[cDType];
  int64_t eachCoreHandNum = shapeSize / tilingPara["useCoreNum"];
  int64_t eachBlock = 32 / dataTypeSize;
  int64_t outLoopNum = 2;
  if (eachCoreHandNum < eachBlock) {
    outLoopNum = 1;
  }

  int64_t outLoopEleNum = eachCoreHandNum / outLoopNum;
  int64_t ubTensorNum = 21;
  if (!maskInput) {
    ubTensorNum = TOTAL_INPUT_NUMS_NO_MASK;
  }
  int64_t totalNumEachCore = ubTensorNum * eachCoreHandNum;
  GELOGD("op [%s] Enter DynamicLSTMGradCellTilling check input pass4", op_type.c_str());
  int64_t innerLoopNum = 0;
  int64_t innerLoopEleNum = 0;
  int64_t lastLoopEleNum = outLoopEleNum;
  int64_t ubTensorSize = outLoopEleNum;
  if (totalNumEachCore * dataTypeSize >= ubByteSize) {
    ubTensorSize = ubByteSize / dataTypeSize / ubTensorNum / outLoopNum / eachBlock * eachBlock;
    innerLoopEleNum = ubTensorSize;
    innerLoopNum = outLoopEleNum / innerLoopEleNum;
    lastLoopEleNum = outLoopEleNum % innerLoopEleNum;
  }
  tilingPara["eleEachCore"] = eachCoreHandNum;
  tilingPara["outLoopNum"] = outLoopNum;
  tilingPara["outLoopEleNum"] = outLoopEleNum;
  tilingPara["innerLoopNum"] = innerLoopNum;
  tilingPara["innerLoopEleNum"] = innerLoopEleNum;
  tilingPara["lastLoopEleNum"] = lastLoopEleNum;
  tilingPara["ubSize"] = ubTensorSize;
  GELOGD("op [%s] Enter DynamicLSTMGradCellTilling check input pass5", op_type.c_str());

  // set tSize
  tilingPara["tSize"] = op_paras.inputs[2].tensor[0].shape[0];
  PrintParams(tilingPara);
  run_info.block_dim = tilingPara["useCoreNum"];
  setRunInfo(tilingPara, run_info);
  std::vector <int64_t> workspace;
  run_info.workspaces = workspace;
  GELOGI("op[%s] tiling run success.", op_type.c_str());
  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(DynamicLSTMGradCell, DynamicLSTMGradCellTiling);
}
// namespace optiling
