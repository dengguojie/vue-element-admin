/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file conv2d_backprop_filter.cpp
 * \brief tiling function of conv2d_backprop_filter
 */
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "cube_tiling_new.h"
#include "graph/debug/ge_log.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "external/graph/operator.h"
#include "op_tiling.h"
#include "op_log.h"

namespace optiling {
const size_t kConv2dDwInputSizeLimit = 3;

/*
 * @brief: tiling function of conv2d_backprop_filter
 * @param [in] op_type: op_type of the conv2d_backprop_filter
 * @param [in] op_paras: inputs/outputs/atts of the conv2d_backprop_filter
 * @param [in] op_compile_info: compile time generated info of the conv2d_backprop_filter
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool Conv2DBpFilterTiling(const std::string& opType, const ge::Operator& opParas, const nlohmann::json& opCompileInfo,
                          utils::OpRunInfo& runInfo) {
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(opParas);
  if (op_desc == nullptr) {
    GELOGE(ge::FAILED, "the op_desc is nullptr.");
    return false;
  }
  ge::ConstGeTensorDescPtr tensor_a_desc = op_desc->GetInputDescPtr(0);
  // the tensor b's index is 2
  ge::ConstGeTensorDescPtr tensor_b_desc = op_desc->GetInputDescPtr(2);
  const ge::GeShape &tensor_a_shape = tensor_a_desc->GetShape();
  const ge::GeShape &tensor_b_shape = tensor_b_desc->GetShape();
  size_t shape_a_dimnum = tensor_a_shape.GetDimNum();
  bool unvalid_size = opParas.GetInputsSize() < kConv2dDwInputSizeLimit || opParas.GetOutputsSize() == 0 ||
                      shape_a_dimnum < kConv2dDimNumLimit || tensor_b_shape.GetDimNum() < kConv2dDimNumLimit;
  if (unvalid_size) {
    GELOGE(ge::FAILED, "the size is unvalid.");
    return false;
  }
  GELOGD("Current format is %s, Ori format is %s",
         ge::TypeUtils::FormatToSerialString(tensor_a_desc->GetFormat()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_a_desc->GetOriginFormat()).c_str());

  try {
    if (opCompileInfo.empty()) {
      GELOGD("op compile info is empty");
      return false;
    }
    // accurate build has only one item, fuzzy build has multiple items
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
        for (auto &key : key_list) {
          auto &item_key = item[key];
          if (item_key.is_object() && !item_key.empty()) {
            std::vector<int32_t> list_value = item_key.begin().value().get<std::vector<int32_t>>();
            opInfo[key][item_key.begin().key()] = list_value;
          }
        }
        std::string key_int = "block_dim";
        auto &item_key_int = item[key_int];
        if (item_key_int.is_object() && !item_key_int.empty()) {
          int32_t int_value = item_key_int.begin().value().get<int32_t>();
          opInfo[key_int][item_key_int.begin().key()] = int_value;
        }
      }
      // <<< end: put together compile info
      GELOGD("compile info after splice is: %s", opInfo.dump().c_str());
    } else if (opCompileInfo.is_object()) {
      varMap = opCompileInfo.at("_vars")["10000"].get<std::vector<std::string>>();
      opInfo = opCompileInfo;
    }

    std::vector<int64_t> var_value;
    if (std::find(varMap.begin(), varMap.end(), "batch") != varMap.end()) {
      var_value.insert(var_value.end(), tensor_a_shape.GetDim(kConv2dNDim));
    }
    if (std::find(varMap.begin(), varMap.end(), "fmap_h") != varMap.end()) {
      var_value.insert(var_value.end(), tensor_a_shape.GetDim(kConv2dHDim));
      var_value.insert(var_value.end(), tensor_b_shape.GetDim(kConv2dHDim));
    }
    if (std::find(varMap.begin(), varMap.end(), "fmap_w") != varMap.end()) {
      var_value.insert(var_value.end(), tensor_a_shape.GetDim(kConv2dWDim));
      var_value.insert(var_value.end(), tensor_b_shape.GetDim(kConv2dWDim));
    }

    std::vector<int64_t> input_shape;
    input_shape.reserve(shape_a_dimnum);
    for (size_t i = 0; i < shape_a_dimnum; i++) {
      input_shape.emplace_back(tensor_a_shape.GetDim(i));
    }
    return cube_tiling(opType, input_shape, var_value, opInfo, runInfo);
  } catch (...) {
    GELOGD("get unknown exception, please check compile info json.");
    return false;
  }
}

// register tiling interface of the conv2d_backprop_filter
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Conv2DBackpropFilter, Conv2DBpFilterTiling);
REGISTER_OP_TILING_FUNC_BUFFERED_V2(DepthwiseConv2DBackpropFilter, Conv2DBpFilterTiling);
}  // namespace optiling
