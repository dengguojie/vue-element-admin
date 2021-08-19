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
 * \file gather_v2.cpp
 * \brief tiling function of op
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

const int64_t BLOCK_SIZE = 32;
const int64_t PARAMS_CACHED_UB = 100 * 1024;
const int64_t RESERVED_UB_SIZE = 6 * 1024;

// A: params larger than cache_ub
// B: indices larger than the number contained in one block for each core
// C: remaining indices larger than one block

// A
const int64_t TILING_MODE_1 = 1;
// B
const int64_t TILING_MODE_4 = 4;

// 
const int64_t TILING_MODE_2 = 2;
// B C
const int64_t TILING_MODE_5 = 5;

// A B
const int64_t TILING_MODE_3 = 3;
// A B C
const int64_t TILING_MODE_6 = 6;

struct GatherElementsTilingParams {
  int64_t tilingMode;
  int64_t paramsPre;
  int64_t paramsAxis;
  int64_t paramsRow;
  int64_t indicesNum;
  int64_t need_core_num;
  int64_t indices_num_each_core;
  int64_t indices_num_remaining;
  int64_t indices_loop_num;
  int64_t indices_row_num_once;
  int64_t indices_row_num_last;
  int64_t paramsTotal;
  int64_t remaining_block_remain;
  int64_t remaining_block_num;
};

void InitGatherElementsParams(GatherElementsTilingParams& params) {
  params.tilingMode = 0;
  params.paramsPre = 1;
  params.paramsAxis = 1;
  params.paramsRow = 1;
  params.indicesNum = 1;
  params.need_core_num = 0;
  params.indices_num_each_core = 0;
  params.indices_num_remaining = 0;
  params.indices_loop_num = 0;
  params.indices_row_num_once = 0;
  params.indices_row_num_last = 0;
  params.paramsTotal = 0;
  params.remaining_block_remain = 0;
  params.remaining_block_num = 0;
}

void SetGatherElementsParams(GatherElementsTilingParams& Params, OpRunInfo& runInfo) {
  // set tiling data
  ByteBufferPut(runInfo.tiling_data, Params.tilingMode); //1
  ByteBufferPut(runInfo.tiling_data, Params.paramsPre); //2
  ByteBufferPut(runInfo.tiling_data, Params.paramsAxis); //3
  ByteBufferPut(runInfo.tiling_data, Params.paramsRow); //4
  ByteBufferPut(runInfo.tiling_data, Params.indicesNum); //5
  ByteBufferPut(runInfo.tiling_data, Params.need_core_num); //6
  ByteBufferPut(runInfo.tiling_data, Params.indices_num_each_core); //7
  ByteBufferPut(runInfo.tiling_data, Params.indices_num_remaining); //8
  ByteBufferPut(runInfo.tiling_data, Params.indices_loop_num); //9
  ByteBufferPut(runInfo.tiling_data, Params.indices_row_num_once); //10
  ByteBufferPut(runInfo.tiling_data, Params.indices_row_num_last); //11
  ByteBufferPut(runInfo.tiling_data, Params.paramsTotal); //12
  ByteBufferPut(runInfo.tiling_data, Params.remaining_block_remain); //13
  ByteBufferPut(runInfo.tiling_data, Params.remaining_block_num); //14
}

void PrintGatherElementsParams(const GatherElementsTilingParams& params) {
  GELOGD("op [GatherElementsTiling] : tilingMode=%d.", params.tilingMode);
  GELOGD("op [GatherElementsTiling] : paramsPre=%d.", params.paramsPre);
  GELOGD("op [GatherElementsTiling] : paramsAxis=%d.", params.paramsAxis);
  GELOGD("op [GatherElementsTiling] : paramsRow=%d.", params.paramsRow);
  GELOGD("op [GatherElementsTiling] : indicesNum=%d.", params.indicesNum);
  GELOGD("op [GatherElementsTiling] : need_core_num=%d.", params.need_core_num);
  GELOGD("op [GatherElementsTiling] : indices_num_each_core=%d.", params.indices_num_each_core);
  GELOGD("op [GatherElementsTiling] : indices_num_remaining=%d.", params.indices_num_remaining);
  GELOGD("op [GatherElementsTiling] : indices_loop_num=%d.", params.indices_loop_num);
  GELOGD("op [GatherElementsTiling] : indices_row_num_once=%d.", params.indices_row_num_once);
  GELOGD("op [GatherElementsTiling] : indices_row_num_last=%d.", params.indices_row_num_last);
  GELOGD("op [GatherElementsTiling] : paramsTotal=%d.", params.paramsTotal);
  GELOGD("op [GatherElementsTiling] : remaining_block_remain=%d.", params.remaining_block_remain);
  GELOGD("op [GatherElementsTiling] : remaining_block_num=%d.", params.remaining_block_num);
}

bool checkTensorShape(const std::string& opType, std::vector<int64_t> indicesShape, std::vector<int64_t> yShape) {
  int64_t indicesDims = indicesShape.size();
  int64_t yDims = yShape.size();

  std::vector<int64_t> outputShape;
  for (int64_t i = 0; i < indicesDims; i++) {
    outputShape.push_back(indicesShape[i]);
  }

  int64_t outputDims = outputShape.size();

  if (yDims != outputDims) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "y", "the dim of y must be equal to the dim of output");
    OP_LOGE(opType.c_str(), "op [GatherElementsTiling] : CheckTensorShape, y Shape is invalid.");
    return false;
  }

  for (int64_t i = 0; i < yDims; i++) {
    if (yShape[i] != outputShape[i]) {
      ge::OpsOneInputShapeErrReport(opType.c_str(), "y", "the shape of y must be equal to the shape of output");
      OP_LOGE(opType.c_str(), "op [GatherElementsTiling] : CheckTensorShape, y Shpae dim is invalid.");
      return false;
    }
  }

  return true;
}

bool GetCompileParams(const std::string& opType, const nlohmann::json& opCompileInfoJson, int64_t& coreNum,
                      int64_t& ubSize, int64_t& l1Size, int64_t& paramsDSize, int64_t& indicesDSize, int64_t& axis) {
  using namespace nlohmann;

  const auto& allVars = opCompileInfoJson["vars"];
  if (allVars.count("core_num") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "core_num");
    OP_LOGE(opType.c_str(), "op [GatherElementsTiling] : GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();
  if (allVars.count("ub_size") == 0) {
    OP_LOGE(opType.c_str(), "op [GatherElementsTiling] : GetCompileParams, get ub_size error");
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "ub_size");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int64_t>();
  if (allVars.count("l1_size") == 0) {
    OP_LOGE(opType.c_str(), "op [GatherElementsTiling] : GetCompileParams, get l1_size error");
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "l1_size");
    return false;
  }
  l1Size = allVars["l1_size"].get<std::int64_t>();
  if (allVars.count("params_dsize") == 0) {
    OP_LOGE(opType.c_str(), "op [GatherElementsTiling] : GetCompileParams, get params_dsize error");
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "params_dsize");
    return false;
  }
  paramsDSize = allVars["params_dsize"].get<std::int64_t>();
  if (allVars.count("indices_dsize") == 0) {
    OP_LOGE(opType.c_str(), "op [GatherElementsTiling] : GetCompileParams, get indices_dsize error");
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "indices_dsize");
    return false;
  }
  indicesDSize = allVars["indices_dsize"].get<std::int64_t>();
  if (allVars.count("axis") == 0) {
    OP_LOGE(opType.c_str(), "op [GatherElementsTiling] : GetCompileParams, get axis error");
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "axis");
    return false;
  }
  axis = allVars["axis"].get<std::int64_t>();
  return true;
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool GatherElementsTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                          OpRunInfo& runInfo) {
  GELOGI("op[%s] GatherElementsTiling running.", opType.c_str());
  using namespace ge;
  if (op_info == nullptr) {
    OP_LOGE(opType.c_str(), "op GatherElementsTiling: op_info json error.");
    return false;
  }
  if (opParas.inputs.empty() || opParas.inputs.size() < 2 || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "x or indices",
                                  "The length of inputs is less than 3 or the inputs is empty");
    OP_LOGE(opType.c_str(), "op GatherElementsTiling: input shape error.");
    return false;
  }
  if (opParas.outputs.empty() || opParas.outputs.size() < 1 || opParas.outputs[0].tensor.empty()) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "y", "The length of outputs is less than 1 or the outputs is empty");
    OP_LOGE(opType.c_str(), "op GatherElementsTiling: output shape error.");
    return false;
  }

  std::vector<int64_t> paramsShape = opParas.inputs[0].tensor[0].shape;
  std::vector<int64_t> indicesShape = opParas.inputs[1].tensor[0].shape;
  std::vector<int64_t> indicesOriShape = opParas.inputs[1].tensor[0].ori_shape;
  std::vector<int64_t> yShape = opParas.outputs[0].tensor[0].shape;

  // check inputs shape
  int64_t paramsDims = paramsShape.size();
  int64_t indicesDims = indicesShape.size();
  if (paramsDims <= 0 || indicesDims <= 0) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "x or indices", "the dim of x or indices is less than 1");
    OP_LOGE("op[%s] GatherElementsTiling: paramsDims or indicesDims is 0.", opType.c_str());
    return false;
  }

  // get compile info
  int64_t ubSize = 0;
  int64_t l1Size = 0;
  int64_t coreNum = 0;
  int64_t paramsDSize = 0;
  int64_t indicesDSize = 0;
  int64_t axis = 0;
  
  bool flag = GetCompileParams(opType, op_info, coreNum, ubSize, l1Size, paramsDSize, indicesDSize, axis);

  if (!flag) {
    OP_LOGE("op[%s] GatherElementsTiling: GetCompileParams error.", opType.c_str());
    return false;
  }

  if (axis < -paramsDims || axis >= paramsDims) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "axis",
                                  "the dim of axis is less than negative x dim, or greater than x dim");
    OP_LOGE(opType.c_str(), "op GatherElementsTiling: axis is invalid.");
    return false;
  }
  if (axis < 0) {
    axis += paramsDims;
  }
  
  bool ret = checkTensorShape(opType, indicesShape, yShape);
  if (!ret) {
    OP_LOGE(opType.c_str(), "op GatherElementsTiling: [checkTensorShape] failed.");
    return ret;
  }

  GatherElementsTilingParams runParams;
  InitGatherElementsParams(runParams);

  // params shape convert to 3D:[paramsPre, paramsAxis, paramsRow]
  // indices shape convert to 1D:[indicesNum]
  // output tensor, y shape convert to:[paramsPre, indicesNum, paramsRow]
  if (axis == 0) {
    runParams.paramsPre = 1;
  } else {
    for (int i = 0; i < axis; i++) {
      runParams.paramsPre *= paramsShape[i];
    }
  }
  runParams.paramsAxis = paramsShape[axis];
  if (axis + 1 < paramsDims) {
    for (int i = axis + 1; i < paramsDims; i++) {
      runParams.paramsRow *= paramsShape[i];
    }
  } else {
    runParams.paramsRow = 1;
  }

  int64_t availableUbSize = ubSize - 2 * 1024;  // reserved 2K
  int64_t halfUbSize = availableUbSize / 2;
  int64_t paramsBlockNum = BLOCK_SIZE / paramsDSize;
  int64_t indicesBlockNum = BLOCK_SIZE / indicesDSize;

  runParams.paramsTotal = std::accumulate(paramsShape.begin(), paramsShape.end(), 1, std::multiplies<int64_t>());
  int64_t paramsTotalCeil = (runParams.paramsTotal + paramsBlockNum - 1) / paramsBlockNum * paramsBlockNum;

  for (int i = 0; i < indicesDims; i++) {
    runParams.indicesNum *= indicesShape[i];
  }

  int64_t halfUbIndicesElem = halfUbSize / indicesDSize;

  if (runParams.indicesNum >= indicesBlockNum * coreNum){
    runParams.need_core_num = coreNum;
    runParams.indices_num_each_core = runParams.indicesNum / runParams.need_core_num;
    runParams.indices_num_remaining = runParams.indicesNum % runParams.need_core_num;
    runParams.indices_loop_num = runParams.indices_num_each_core / halfUbIndicesElem;
    runParams.indices_row_num_once = halfUbIndicesElem;
    runParams.indices_row_num_last = runParams.indices_num_each_core % runParams.indices_row_num_once;

    if (paramsTotalCeil >= PARAMS_CACHED_UB / paramsDSize){
      if (indicesBlockNum >= runParams.indices_num_remaining){
        runParams.tilingMode = TILING_MODE_3;
      }
      else{
        runParams.tilingMode = TILING_MODE_6;
        runParams.remaining_block_remain = runParams.indices_num_remaining % indicesBlockNum;
        runParams.remaining_block_num = runParams.indices_num_remaining / indicesBlockNum;
      }
    }
    else {
      if (indicesBlockNum >= runParams.indices_num_remaining){
        runParams.tilingMode = TILING_MODE_4;
      }
      else{
        runParams.tilingMode = TILING_MODE_5;
        runParams.remaining_block_remain = runParams.indices_num_remaining % indicesBlockNum;
        runParams.remaining_block_num = runParams.indices_num_remaining / indicesBlockNum;
      }
    }

  }
  else {
    runParams.need_core_num = 1;
    runParams.indices_num_each_core = runParams.indicesNum;
    runParams.indices_num_remaining = 0;
    runParams.indices_loop_num = 0;
    runParams.indices_row_num_once = 0;
    runParams.indices_row_num_last = runParams.indicesNum;

    if (paramsTotalCeil >= PARAMS_CACHED_UB / paramsDSize){
       runParams.tilingMode = TILING_MODE_1;
    }
    else {
      runParams.tilingMode = TILING_MODE_2;
    }    
  }

  SetGatherElementsParams(runParams, runInfo);
  PrintGatherElementsParams(runParams);

  // block_dim, core num used in tik op
  runInfo.block_dim = runParams.need_core_num;
  // workspace, null for tik op
  std::vector<int64_t> workspace;
  workspace.push_back(2147483647);
  runInfo.workspaces = workspace;
  GELOGI("op[%s] tiling run success.", opType.c_str());

  return true;
}

// register tiling interface of the GatherElements op.
REGISTER_OP_TILING_FUNC_BUFFERED(GatherElements, GatherElementsTiling);

}  // namespace optiling
