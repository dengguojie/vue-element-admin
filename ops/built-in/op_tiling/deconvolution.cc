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
 * \file deconvolution.cpp
 * \brief tiling function of deconvolution
 */
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "cube_tiling_new.h"
#include "external/graph/operator.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_tiling.h"

namespace optiling {
/*
 * @brief: tiling function of deconvlution
 * @param [in] op_type: op_type of deconvlution
 * @param [in] op_paras: inputs/outputs/atts of deconvlution
 * @param [in] op_compile_info: compile time generated info of deconvlution
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool DeConvlutionTiling(const std::string& opType, const ge::Operator& opParas, const nlohmann::json& opCompileInfo,
                        utils::OpRunInfo& runInfo) {
  int32_t nDim = 0;
  int32_t hDim = 2;
  int32_t wDim = 3;

  if (opParas.GetInputsSize() == 0 || opParas.GetOutputsSize() == 0 ||
      opParas.GetInputDesc(0).GetShape().GetDimNum() == 0 || opParas.GetOutputDesc(0).GetShape().GetDimNum() == 0){
    return false;
  }

  auto output_format = ge::TypeUtils::FormatToSerialString(opParas.GetOutputDesc(0).GetFormat()).c_str();
  auto output_ori_format = ge::TypeUtils::FormatToSerialString(opParas.GetOutputDesc(0).GetOriginFormat()).c_str();
  GELOGD("Current format is %s, Ori format is %s", output_format, output_ori_format);

  if (opCompileInfo.empty()) {
    GELOGD("op compile info is empty");
    return false;
  }
  // accurate build has only one item
  // fuzzy build has multiple items
  std::vector<std::string> varMap;
  nlohmann::json opInfo;
  GELOGD("original compile info is: %s", opCompileInfo.dump().c_str());
  if (opCompileInfo.is_array()) {
    // >>> start: splice compile info
    opInfo = opCompileInfo[0];
    varMap = opInfo.at("_vars").begin().value().get<std::vector<std::string>>();
    nlohmann::json item;
    for (size_t i = 1; i < opCompileInfo.size(); ++i) {
      item = opCompileInfo[i];
      std::vector<std::string> key_list = {"repo_seeds", "repo_range", "cost_range"};
      for (auto key: key_list) {
        if (item[key].is_object() && !item[key].empty()) {
          std::vector<int32_t> list_value = item[key].begin().value().get<std::vector<int32_t>>();
          opInfo[key][item[key].begin().key()] = list_value;
        }
      }
      std::vector<std::string> key_int = {"block_dim"};
      for (auto key: key_int) {
        if (item[key].is_object() && !item[key].empty()) {
          int32_t int_value = item[key].begin().value().get<int32_t>();
          opInfo[key][item[key].begin().key()] = int_value;
        }
      }
    }
    // <<< end: put together compile info
    GELOGD("compile info after splice is: %s", opInfo.dump().c_str());
  } else if (opCompileInfo.is_object()) {
    varMap = opCompileInfo.at("_vars")["10000"].get<std::vector<std::string>>();
    opInfo = opCompileInfo;
  }

  std::vector<int64_t> var_value;
  if (std::find(varMap.begin(), varMap.end(), "batch_n") != varMap.end()) {
    var_value.insert(var_value.end(), opParas.GetOutputDesc(0).GetShape().GetDim(nDim));
  }
  if (std::find(varMap.begin(), varMap.end(), "dx_h") != varMap.end()) {
    var_value.insert(var_value.end(), opParas.GetInputDesc(0).GetShape().GetDim(hDim));
    var_value.insert(var_value.end(), opParas.GetOutputDesc(0).GetShape().GetDim(hDim));
  }
  if (std::find(varMap.begin(), varMap.end(), "dx_w") != varMap.end()) {
    var_value.insert(var_value.end(), opParas.GetInputDesc(0).GetShape().GetDim(wDim));
    var_value.insert(var_value.end(), opParas.GetOutputDesc(0).GetShape().GetDim(wDim));
  }

  return cube_tiling(opType, opParas.GetOutputDesc(0).GetShape().GetDims(), var_value, opInfo, runInfo);
}

// register tiling interface of the deconvlution
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Deconvolution, DeConvlutionTiling);
}  // namespace optiling
