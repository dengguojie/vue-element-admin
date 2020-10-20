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
 * \file gather.cpp
 * \brief
 */
#include <string>

#include <nlohmann/json.hpp>
#include "register/op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"

namespace optiling {

const int32_t BLOCK_SIZE = 32;

// 1. one params row size is smaller than 32B
// params is not cache
const int32_t TILING_MODE_1 = 1;
// 2. one params row size is greater than or equal to 32B
// paramsRow is not 32B aligned
const int32_t TILING_MODE_2 = 2;
// paramsRow is 32B aligned
const int32_t TILING_MODE_3 = 3;

bool GatherCheckTensorShape(const std::string& opType, std::vector<int64_t> paramsShape,
                            std::vector<int64_t> indicesShape, std::vector<int64_t> yShape, int32_t axis) {
  int32_t paramsDims = paramsShape.size();
  int32_t indicesDims = indicesShape.size();
  int32_t yDims = yShape.size();

  std::vector<int64_t> outputShape;
  if (axis > 0) {
    for (int32_t i = 0; i < axis; i++) {
      outputShape.push_back(paramsShape[i]);
    }
  }
  for (int32_t i = 0; i < indicesDims; i++) {
    outputShape.push_back(indicesShape[i]);
  }
  if (axis + 1 < paramsDims) {
    for (int32_t i = axis + 1; i < paramsDims; i++) {
      outputShape.push_back(paramsShape[i]);
    }
  }
  int32_t outputDims = outputShape.size();

  if (yDims != outputDims) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "y", "the dim of y must be equal to the dim of output");
    GE_LOGE("op [GatherTiling] : GatherCheckTensorShape, y Shape is invalid.");
    return false;
  }

  for (int32_t i = 0; i < yDims; i++) {
    if (yShape[i] != outputShape[i]) {
      ge::OpsOneInputShapeErrReport(opType.c_str(), "y", "the shape of y must be equal to the shape of output");
      GE_LOGE("op [GatherTiling] : GatherCheckTensorShape, y Shpae dim is invalid.");
      return false;
    }
  }

  return true;
}

bool GatherCompileParams(const std::string& opType, const nlohmann::json& opCompileInfoJson, int32_t& coreNum,
                         int32_t& ubSize, int32_t& l1Size, int32_t& paramsDSize, int32_t& indicesDSize) {
  using namespace nlohmann;

  const auto& allVars = opCompileInfoJson["vars"];
  if (allVars.count("core_num") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "core_num");
    GE_LOGE("op [GatherTiling] : GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int32_t>();
  if (allVars.count("ub_size") == 0) {
    GE_LOGE("op [GatherTiling] : GetCompileParams, get ub_size error");
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "ub_size");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int32_t>();
  if (allVars.count("l1_size") == 0) {
    GE_LOGE("op [GatherTiling] : GetCompileParams, get l1_size error");
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "l1_size");
    return false;
  }
  l1Size = allVars["l1_size"].get<std::int32_t>();
  if (allVars.count("params_dsize") == 0) {
    GE_LOGE("op [GatherTiling] : GetCompileParams, get params_dsize error");
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "params_dsize");
    return false;
  }
  paramsDSize = allVars["params_dsize"].get<std::int32_t>();
  if (allVars.count("indices_dsize") == 0) {
    GE_LOGE("op [GatherTiling] : GetCompileParams, get indices_dsize error");
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "indices_dsize");
    return false;
  }
  indicesDSize = allVars["indices_dsize"].get<std::int32_t>();

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
bool GatherTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                  OpRunInfo& runInfo) {
  GELOGI("op[%s] GatherTiling running.", opType.c_str());
  using namespace ge;
  if (op_info == nullptr) {
    GE_LOGE("op[%s] GatherTiling: op_info json error.", opType.c_str());
    return false;
  }
  if (opParas.inputs.empty() || opParas.inputs.size() < 2 || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "x or indices",
                                  "The length of inputs is less than 2 or the inputs is empty");
    GE_LOGE("op[%s] GatherTiling: input shape error.", opType.c_str());
    return false;
  }
  if (opParas.outputs.empty() || opParas.outputs.size() < 1 || opParas.outputs[0].tensor.empty()) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "y", "The length of outputs is less than 1 or the outputs is empty");
    GE_LOGE("op[%s] GatherTiling: output shape error.", opType.c_str());
    return false;
  }

  std::vector<int64_t> paramsShape = opParas.inputs[0].tensor[0].shape;
  std::vector<int64_t> indicesShape = opParas.inputs[1].tensor[0].shape;
  // std::vector<int64_t> axisShape = opParas.inputs[2].tensor[0].shape;
  std::vector<int64_t> yShape = opParas.outputs[0].tensor[0].shape;

  // if (opParas.const_inputs.find("axis") == opParas.const_inputs.end()){
  //     ge::OpsOneInputShapeErrReport(opType.c_str(), "axis",
  //         "axis is not exist in const inputs");
  //     GE_LOGE("op[%s] GatherTiling: axis not exists.", opType.c_str());
  //     return false;
  // }
  // const int32_t* axis_ptr = reinterpret_cast<const int32_t*>(std::get<0>(opParas.const_inputs.at("axis")));
  int32_t axis = 0;
  GELOGD("op [GatherTiling] : axis=%d.", axis);

  // check inputs shape
  int32_t paramsDims = paramsShape.size();
  int32_t indicesDims = indicesShape.size();
  if (paramsDims <= 0 || indicesDims <= 0) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "x or indices", "the dim of x or indices is less than 1");
    GE_LOGE("op[%s] GatherTiling: paramsDims or indicesDims is 0.", opType.c_str());
    return false;
  }
  if (axis < -paramsDims || axis >= paramsDims) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "axis",
                                  "the dim of axis is less than negative x dim, or greater than x dim");
    GE_LOGE("op[%s] GatherTiling: axis is invalid.", opType.c_str());
    return false;
  }
  if (axis < 0) {
    axis += paramsDims;
  }

  bool ret = GatherCheckTensorShape(opType, paramsShape, indicesShape, yShape, axis);
  if (!ret) {
    GE_LOGE("op[%s] GatherTiling: [GatherCheckTensorShape] failed.", opType.c_str());
    return ret;
  }

  // get compile info
  int32_t ubSize = 0;
  int32_t l1Size = 0;
  int32_t coreNum = 0;
  int32_t paramsDSize = 0;
  int32_t indicesDSize = 0;
  bool flag = GatherCompileParams(opType, op_info, coreNum, ubSize, l1Size, paramsDSize, indicesDSize);
  if (!flag) {
    GE_LOGE("op[%s] GatherTiling: GatherCompileParams error.", opType.c_str());
    return false;
  }

  int32_t availableUbSize = ubSize - 2 * 1024;  // reserved 2K
  int32_t halfUbSize = availableUbSize / 2;

  // params shape convert to 3D:[paramsPre, paramsAxis, paramsRow]
  // indices shape convert to 1D:[indicesNum]
  // output tensor, y shape convert to:[paramsPre, indicesNum, paramsRow]
  int32_t paramsPre = 1, paramsAxis = 1, paramsRow = 1;
  int32_t indicesNum = 1;

  // now only support axis is 0
  if (axis == 0) {
    paramsPre = 1;
  } else {
    for (int i = 0; i < axis; i++) {
      paramsPre *= paramsShape[i];
    }
  }
  paramsAxis = paramsShape[axis];
  if (axis + 1 < paramsDims) {
    for (int i = axis + 1; i < paramsDims; i++) {
      paramsRow *= paramsShape[i];
    }
  } else {
    paramsRow = 1;
  }

  for (int i = 0; i < indicesDims; i++) {
    indicesNum *= indicesShape[i];
  }

  // params tiling mode
  int32_t tilingMode;
  // run parameters compute
  int32_t need_core_num = 0;
  int32_t tail_process_core = 0;
  int32_t indices_num_each_core = 0;
  int32_t indices_num_remaining = 0;
  int32_t indices_loop_num = 0;
  int32_t indices_row_num_once = 0;
  int32_t indices_row_num_last = 0;
  int32_t row_num_once_ub = 0;
  int32_t row_num_once_tail_ub = 0;
  int32_t inner_loop_num = 0;
  int32_t row_num_last_tail_ub = 0;
  int32_t row_num_last_ub = 0;
  int32_t inner_loop_num_last = 0;

  // block tiling: indices tiling
  need_core_num = coreNum;
  tail_process_core = 0;
  indices_num_each_core = indicesNum / need_core_num;
  indices_num_remaining = indicesNum % need_core_num;
  if (indicesNum <= need_core_num) {
    need_core_num = indicesNum;
    tail_process_core = 0;
    indices_num_each_core = 1;
    indices_num_remaining = 0;
  }

  int32_t cacheParams = 0;
  int32_t resUbSize = halfUbSize;  // store params row data
  int32_t halfUbIndicesElem = halfUbSize / indicesDSize;

  // one params row size is smaller than 32B
  if (paramsRow * paramsDSize < BLOCK_SIZE) {
    tilingMode = TILING_MODE_1;

    if ((paramsRow < BLOCK_SIZE) && indices_num_each_core * paramsRow * paramsDSize <= BLOCK_SIZE) {
      need_core_num = 1;
      tail_process_core = 0;
      indices_num_each_core = indicesNum;
      indices_num_remaining = 0;
    }

    indices_loop_num = indices_num_each_core / halfUbIndicesElem;
    indices_row_num_once = halfUbIndicesElem;
    if (indices_num_each_core % indices_row_num_once != 0) {
      indices_row_num_last = indices_num_each_core % indices_row_num_once;
    }

    int32_t blockNum = BLOCK_SIZE / paramsDSize;
    row_num_once_ub = resUbSize / (paramsRow * paramsDSize);
    if (int(row_num_once_ub % blockNum) != 0) {
      row_num_once_ub = int(row_num_once_ub / blockNum) * blockNum;
    }
    inner_loop_num = indices_row_num_once / row_num_once_ub;
    if (indices_row_num_once % row_num_once_ub != 0) {
      row_num_once_tail_ub = indices_row_num_once % row_num_once_ub;
    }
    if (inner_loop_num > 0 && row_num_once_tail_ub > 0 && row_num_once_tail_ub * paramsRow < blockNum) {
      inner_loop_num = inner_loop_num - 1;
      row_num_once_tail_ub = row_num_once_tail_ub + row_num_once_ub;
    }

    row_num_last_ub = resUbSize / (paramsRow * paramsDSize);
    if (int(row_num_last_ub % blockNum) != 0) {
      row_num_last_ub = int(row_num_last_ub / blockNum) * blockNum;
    }
    inner_loop_num_last = indices_row_num_last / row_num_last_ub;
    if (indices_row_num_last % row_num_last_ub != 0) {
      row_num_last_tail_ub = indices_row_num_last % row_num_last_ub;
    }
    if (inner_loop_num_last > 0 && row_num_last_tail_ub > 0 && row_num_last_tail_ub * paramsRow < blockNum) {
      inner_loop_num_last = inner_loop_num_last - 1;
      row_num_last_tail_ub = row_num_last_tail_ub + row_num_once_ub;
    }
  } else {                                            // one params row size is greater than or equal to 32B
    if (paramsRow * paramsDSize % BLOCK_SIZE != 0) {  // not 32B aligned
      tilingMode = TILING_MODE_2;

      GE_LOGE("op[%s] GatherTiling: inputs shape is not support now, x row size is not 32 bytes align.",
              opType.c_str());
      return false;
    } else {  // 32B aligned
      tilingMode = TILING_MODE_3;

      indices_loop_num = indices_num_each_core / halfUbIndicesElem;
      indices_row_num_once = halfUbIndicesElem;
      if (indices_num_each_core % indices_row_num_once != 0) {
        indices_row_num_last = indices_num_each_core % indices_row_num_once;
      }

      row_num_once_ub = resUbSize / (paramsRow * paramsDSize);
      inner_loop_num = indices_row_num_once / row_num_once_ub;
      if (indices_row_num_once % row_num_once_ub != 0) {
        row_num_once_tail_ub = indices_row_num_once % row_num_once_ub;
      }

      row_num_last_ub = resUbSize / (paramsRow * paramsDSize);
      inner_loop_num_last = indices_row_num_last / row_num_last_ub;
      if (indices_row_num_last % row_num_last_ub != 0) {
        row_num_last_tail_ub = indices_row_num_last % row_num_last_ub;
      }
    }
  }

  // set tiling data
  ByteBufferPut(runInfo.tiling_data, tilingMode);

  ByteBufferPut(runInfo.tiling_data, paramsPre);
  ByteBufferPut(runInfo.tiling_data, paramsAxis);
  ByteBufferPut(runInfo.tiling_data, paramsRow);
  ByteBufferPut(runInfo.tiling_data, indicesNum);

  ByteBufferPut(runInfo.tiling_data, cacheParams);
  ByteBufferPut(runInfo.tiling_data, need_core_num);
  ByteBufferPut(runInfo.tiling_data, tail_process_core);
  ByteBufferPut(runInfo.tiling_data, indices_num_each_core);
  ByteBufferPut(runInfo.tiling_data, indices_num_remaining);
  ByteBufferPut(runInfo.tiling_data, indices_loop_num);
  ByteBufferPut(runInfo.tiling_data, indices_row_num_once);
  ByteBufferPut(runInfo.tiling_data, indices_row_num_last);
  ByteBufferPut(runInfo.tiling_data, row_num_once_ub);
  ByteBufferPut(runInfo.tiling_data, row_num_once_tail_ub);
  ByteBufferPut(runInfo.tiling_data, inner_loop_num);
  ByteBufferPut(runInfo.tiling_data, row_num_last_ub);
  ByteBufferPut(runInfo.tiling_data, row_num_last_tail_ub);
  ByteBufferPut(runInfo.tiling_data, inner_loop_num_last);
  GELOGD("op [GatherTiling] : tilingMode=%d, paramsRow=%d", tilingMode, paramsRow);

  // block_dim, core num used in tik op
  runInfo.block_dim = need_core_num;
  // workspace, null for tik op
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  GELOGI("op[%s] tiling run success.", opType.c_str());

  return true;
}

// register tiling interface of the Gather op.
REGISTER_OP_TILING_FUNC(Gather, GatherTiling);

}  // namespace optiling
