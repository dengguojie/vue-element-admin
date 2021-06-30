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
 * \file avg_pool.cc
 * \brief tiling function of avg_pool
 */
#include <map>
#include <string>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "external/graph/operator.h"
#include "cube_tiling.h"
#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "error_log.h"

namespace optiling {
/*
 * @brief: tiling function of avg_pool
 * @param [in] op_type: op_type of the avg_pool
 * @param [in] op_paras: inputs/outputs/atts of the avg_pool
 * @param [in] op_compile_info: compile time generated info of the avg_pool
 * @param [out] run_info: result data
 * @return bool: success or not
 */
using namespace ge;
using namespace std;

const int32_t MAX_STRIDE = 63;

bool AvgPoolV2TilingCube(const std::string& opType, const ge::Operator& opParas, const nlohmann::json& opCompileInfo,
                         utils::OpRunInfo& runInfo) {
  int32_t nDim = 0;
  int32_t hDim = 2;
  int32_t wDim = 3;

  if (opParas.GetInputsSize() == 0 || opParas.GetOutputsSize() == 0 || 
      opParas.GetInputDesc(0).GetShape().GetDimNum() == 0 || opParas.GetOutputDesc(0).GetShape().GetDimNum() == 0) {
    return false;
  }

  if(opCompileInfo.empty()) {
    GELOGD("op compile info is empty");
    return false;
  }
  // accurate build has only one item
  // fuzzy build has multiple items
  GELOGD("original compile info is: %s", opCompileInfo.dump().c_str());

  std::vector<std::string> varMap = opCompileInfo.at("_vars").begin().value().get<std::vector<std::string>>();
  int64_t batch = opParas.GetInputDesc(0).GetShape().GetDim(nDim);
  int64_t hi = opParas.GetInputDesc(0).GetShape().GetDim(hDim);
  int64_t ho = opParas.GetOutputDesc(0).GetShape().GetDim(hDim);
  int64_t wi = opParas.GetInputDesc(0).GetShape().GetDim(wDim);
  int64_t wo = opParas.GetOutputDesc(0).GetShape().GetDim(wDim);
  std::vector<int64_t> var_value;
  for (auto var:varMap) {
    if (var == "batch_n") {
      var_value.insert(var_value.end(), batch);
    } else if (var == "fmap_h") {
      var_value.insert(var_value.end(), hi);
    } else if (var == "ho") {
      var_value.insert(var_value.end(), ho);
    } else if (var == "fmap_w") {
      var_value.insert(var_value.end(), wi);
    } else if (var == "wo") {
      var_value.insert(var_value.end(), wo);
    }
  }
  GELOGD("avgpoolv2 tiling_data is %d, %d, %d, %d, %d, %d", runInfo.GetTilingKey(), batch, hi, ho, wi, wo);
  return cube_tiling1(opType, opParas.GetInputDesc(0).GetShape().GetDims(), var_value, opCompileInfo, runInfo);
}
// register tiling interface of the avgpool
bool AvgPoolTilingV2(const std::string& opType, const ge::Operator& opParas, const nlohmann::json& opCompileInfo,
                     utils::OpRunInfo& runInfo)
{
  int32_t strides_h = opCompileInfo.at("strides_h");
  int32_t strides_w = opCompileInfo.at("strides_w");
  bool result = true;

  if (strides_h <= MAX_STRIDE && strides_w <= MAX_STRIDE) {
    result = AvgPoolV2TilingCube(opType, opParas, opCompileInfo, runInfo);
  } else {
    GELOGW("optiling is not support AvgPoolV2TilingVector");
  }
  return result;
}
REGISTER_OP_TILING_FUNC_BUFFERED_V2(AvgPoolV2, AvgPoolTilingV2);
}  // namespace optiling