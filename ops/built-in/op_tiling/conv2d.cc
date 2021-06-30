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
 * \file conv2d.cpp
 * \brief tiling function of conv2d
 */
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "cube_tiling.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "external/graph/operator.h"
#include "op_tiling.h"
#include "op_log.h"
#include "../op_proto/util/error_util.h"

using namespace std;
namespace optiling {
/*
 * @brief: set val value
 * @param [in] varMap: varMap of conv2d
 * @param [in] op_paras: inputs/outputs/atts of the conv2d
 * @param [out] valValue: val value
 */
std::vector<int64_t> setValValue(std::vector<std::string> varMap, const ge::Operator& opParas) {
  int32_t nDim = 0;
  int32_t hDim = 2;
  int32_t wDim = 3;

  int32_t batch = opParas.GetInputDesc(0).GetShape().GetDim(nDim);
  int32_t hi = opParas.GetInputDesc(0).GetShape().GetDim(hDim);
  int32_t wi = opParas.GetInputDesc(0).GetShape().GetDim(wDim);
  int32_t ho = opParas.GetOutputDesc(0).GetShape().GetDim(hDim);
  int32_t wo = opParas.GetOutputDesc(0).GetShape().GetDim(wDim);
  std::vector<int64_t> varValue;
  for (auto var:varMap) {
    if (var == "batch_n") {
      varValue.insert(varValue.end(), batch);
    } else if (var == "fmap_h") {
      varValue.insert(varValue.end(), hi);
    } else if (var == "fmap_w") {
      varValue.insert(varValue.end(), wi);
    } else if (var == "ho") {
      varValue.insert(varValue.end(), ho);
    } else if (var == "wo") {
      varValue.insert(varValue.end(), wo);
    }
  }
  return varValue;
}
/*
 * @brief: tiling function of conv2d
 * @param [in] op_type: op_type of the conv2d
 * @param [in] op_paras: inputs/outputs/atts of the conv2d
 * @param [in] op_compile_info: compile time generated info of the conv2d
 * @param [out] run_info: result data
 * @return bool: success or not
 */

bool Conv2DTiling(const std::string& opType, const ge::Operator& opParas, const nlohmann::json& opCompileInfo,
                  utils::OpRunInfo& runInfo) {
  int32_t nDim = 0;
  int32_t cDim = 1;
  int32_t hDim = 2;
  int32_t wDim = 3;
  if (opParas.GetInputsSize() == 0 || opParas.GetOutputsSize() == 0 || 
      opParas.GetInputDesc(0).GetShape().GetDimNum() == 0 || opParas.GetOutputDesc(0).GetShape().GetDimNum() == 0) {
    return false;
  }

  if (opType.c_str() == "Conv2D" && opCompileInfo.contains("fmap_c1") && 
      opParas.GetInputDesc(0).GetShape().GetDim(cDim) != opCompileInfo["fmap_c1"]) {
    CUBE_INNER_ERR_REPORT(opType.c_str(), "Not support, input x channel should be equal to filter channel*groups;"
      "x_channel=%d, fmap_c1=%d", opParas.GetInputDesc(0).GetShape().GetDim(cDim), opCompileInfo["fmap_c1"]);
    return false;
  }

  if(opCompileInfo.empty()) {
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

  std::vector<int64_t> varValue = setValValue(varMap, opParas);

  bool res = cube_tiling1(opType, opParas.GetInputDesc(0).GetShape().GetDims(), varValue, opInfo, runInfo);
  // for log
  int32_t batch = opParas.GetInputDesc(0).GetShape().GetDim(nDim);
  int32_t hi = opParas.GetInputDesc(0).GetShape().GetDim(hDim);
  int32_t wi = opParas.GetInputDesc(0).GetShape().GetDim(wDim);
  int32_t ho = opParas.GetOutputDesc(0).GetShape().GetDim(hDim);
  int32_t wo = opParas.GetOutputDesc(0).GetShape().GetDim(wDim);
  GELOGD("tiling_data is %d, %d, %d, %d, %d, %d", runInfo.GetTilingKey(), batch, hi, ho, wi, wo);

  return res;
}

// register tiling interface of the conv2d
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Conv2D, Conv2DTiling);
REGISTER_OP_TILING_FUNC_BUFFERED_V2(DepthwiseConv2D, Conv2DTiling);
}  // namespace optiling
