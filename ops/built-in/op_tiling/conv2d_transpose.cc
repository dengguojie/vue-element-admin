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
 * \file conv2d_transpose.cc
 * \brief tiling function of conv2d_transpose
 */
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "cube_tiling_new.h"
#include "external/graph/operator.h"
#include "graph/debug/ge_log.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "op_tiling.h"
#include "op_log.h"

namespace optiling {
const size_t kConv2dTransInputSizeLimit = 2;

/*
 * @brief: tiling function of conv2d_transpose
 * @param [in] op_type: op_type of conv2d_transpose
 * @param [in] op_paras: inputs/outputs/atts of conv2d_transpose
 * @param [in] op_compile_info: compile time generated info of conv2d_transpose
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool Conv2DTransposeTiling(const std::string& opType, const ge::Operator& opParas, const nlohmann::json& opCompileInfo,
                         utils::OpRunInfo& runInfo) {
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(opParas);
  if (op_desc == nullptr) {
    GELOGE(ge::FAILED, "the op_desc is nullptr.");
    return false;
  }
  // the input tensor's index is 1
  ge::ConstGeTensorDescPtr tensor_in_desc = op_desc->GetInputDescPtr(1);
  ge::ConstGeTensorDescPtr tensor_out_desc = op_desc->GetOutputDescPtr(0);
  const ge::GeShape &tensor_in_shape = tensor_in_desc->GetShape();
  const ge::GeShape &tensor_out_shape = tensor_out_desc->GetShape();
  size_t output_dimnum = tensor_out_shape.GetDimNum();
  bool unvalid_size = opParas.GetInputsSize() < kConv2dTransInputSizeLimit || opParas.GetOutputsSize() == 0 ||
                      tensor_in_shape.GetDimNum() < kConv2dDimNumLimit || output_dimnum < kConv2dDimNumLimit;
  if (unvalid_size) {
    GELOGE(ge::FAILED, "the size is unvalid.");
    return false;
  }
  GELOGD("Current format is %s, Ori format is %s",
         ge::TypeUtils::FormatToSerialString(tensor_out_desc->GetFormat()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_out_desc->GetOriginFormat()).c_str());

  try {
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
    if (std::find(varMap.begin(), varMap.end(), "batch_n") != varMap.end()) {
      var_value.insert(var_value.end(), tensor_out_shape.GetDim(kConv2dNDim));
    }
    if (std::find(varMap.begin(), varMap.end(), "dx_h") != varMap.end()) {
      var_value.insert(var_value.end(), tensor_in_shape.GetDim(kConv2dHDim));
      var_value.insert(var_value.end(), tensor_out_shape.GetDim(kConv2dHDim));
    }
    if (std::find(varMap.begin(), varMap.end(), "dx_w") != varMap.end()) {
      var_value.insert(var_value.end(), tensor_in_shape.GetDim(kConv2dWDim));
      var_value.insert(var_value.end(), tensor_out_shape.GetDim(kConv2dWDim));
    }

    std::vector<int64_t> output_shape;
    output_shape.reserve(output_dimnum);
    for (size_t i = 0; i < output_dimnum; i++) {
      output_shape.emplace_back(tensor_out_shape.GetDim(i));
    }
    return cube_tiling(opType, output_shape, var_value, opInfo, runInfo);
  } catch (...) {
    GELOGD("get unknown exception, please check compile info json.");
    return false;
  }
}

// register tiling interface of the conv2d_transpose
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Conv2DTranspose, Conv2DTransposeTiling);
}  // namespace optiling
