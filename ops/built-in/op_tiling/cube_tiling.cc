/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file cube_tiling.cpp
 * \brief
 */
#include <stdlib.h>
#include <string>
#include "cube_tiling.h"
#include "../op_proto/util/error_util.h"

namespace {
  constexpr int32_t kRangeDivShape = 2;
  constexpr int32_t kConv3dDimSizeLimit = 6;
  constexpr int32_t kConv3dVarDimSizeLimit = 4;
  const string kCompileRepoSeeds = "repo_seeds";
  const string kCompileRepoRange = "repo_range";
  const string kCompileCostRange = "cost_range";
  const std::vector<int32_t> kConv3DDynamicShapeDims = {0, 1, 3, 4}; // format: NDC1HWC0
  const std::vector<int32_t> kConv3DDynamicRangeDims = {0, 1, 2, 3}; // foramt: NDHW

  const std::vector<std::string> kConv3DVarNames = {"batch_n", "fmap_d", "fmap_h", "fmap_w"};
  const std::vector<std::string> kConv3DBpInputVarNames = {"batch_n", "dedy_d", "dedy_h", "dedy_w"};
  const std::vector<std::string> kConv3DTransposeVarNames = {"batch_n", "dedy_d", "dedy_h", "dedy_w"};
  const std::vector<std::string> kAvgPool3DGradVarNames = {"batch_n", "dedy_d", "dedy_h", "dedy_w"};
  const std::map<std::string, std::vector<std::string>> kOpVarNamesMap = {
    {"Conv3D", kConv3DVarNames},
    {"Conv3DBackpropFilter", kConv3DVarNames},
    {"Conv3DBackpropInput", kConv3DBpInputVarNames},
    {"Conv3DTranspose", kConv3DTransposeVarNames},
    {"AvgPool3D", kConv3DVarNames},
    {"AvgPool3DGrad", kAvgPool3DGradVarNames}
  };

  void get_var_names(const std::string &op_type, std::vector<std::string> &vars) {
    if (kOpVarNamesMap.count(op_type) > 0){
      vars = kOpVarNamesMap.at(op_type);
    }
  }

  static bool is_shape_in_range(const std::vector<int64_t> &shape, const std::vector<int64_t> &range) {
    // shape: NDHWC, range: NDHW
    if (shape.size() < kConv3dDimSizeLimit || range.size() < (kConv3dVarDimSizeLimit * 2)) {
      return false;
    }

    for (int32_t i = 0; i < kConv3dVarDimSizeLimit; ++i) {
      int32_t shape_index = kConv3DDynamicShapeDims[i];
      int32_t range_index = kConv3DDynamicRangeDims[i];
      if (shape[shape_index] < range[range_index * kRangeDivShape] || shape[shape_index] > range[(range_index * kRangeDivShape) + 1]) {
        return false;
      }
    }

    return true;
  }

  static bool is_shape_in_range_cube(const std::vector<int64_t> &shape, const std::vector<int64_t> &range) {
    const std::vector<int32_t> shape_dim = {0, 2, 3};
    const std::vector<int32_t> range_dim = {0, 1, 2, 3, 4, 5};
    if (range.size() == range_dim.size()) {
      for (size_t i = 0; i < shape_dim.size(); ++i) {
        if (shape[shape_dim[i]] < range[range_dim[i * kRangeDivShape]] || shape[shape_dim[i]] > range[range_dim[i * kRangeDivShape + 1]]) {
          return false;
        }
      }
    } else if (range.size() == kRangeDivShape) {
      if (shape[shape_dim[0]] < range[0] || shape[shape_dim[0]] > range[1]) {
        return false;
      }
    } else {
      return false;
    }

    return true;
  }

  static int64_t calc_dist(const std::vector<int64_t> &shape, const std::vector<int64_t> &seeds) {
    // shape: NDHWC, range: NDHW
    if (shape.size() < kConv3dDimSizeLimit || seeds.size() < kConv3dVarDimSizeLimit) {
      return std::numeric_limits<int64_t>::max();
    }

    int64_t dist = 0;
    // skip batch
    for (int32_t i = 1; i < kConv3dVarDimSizeLimit; ++i) {
      int32_t shape_index = kConv3DDynamicShapeDims[i];
      int32_t seeds_index = kConv3DDynamicRangeDims[i];
      dist += abs(shape[shape_index] - seeds[seeds_index]);
    }

    return dist;
  }

  std::string get_conv3D_batch_tiling(const std::string& op_type, const std::vector<int64_t>& cur_shape,
                                   const nlohmann::json& compile_info) {
    std::string tiling_id;
    if (cur_shape.empty()) {
      return tiling_id;
    }

    if (!compile_info.contains("tiling_range")) {
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "no tiling_range in compile info json");
      return tiling_id;
    }

    int64_t batch = cur_shape[0];
    auto& tiling_case = compile_info.at("tiling_range");
    for (auto it = tiling_case.begin(); it != tiling_case.end(); ++it) {
      auto& range = it.value();
      if (batch >= range[0] && batch <= range[1]) {
        tiling_id = it.key();
      }
    }

    return tiling_id;
  }

  std::string get_conv3D_ndhw_tiling(const std::string& op_type, const std::vector<int64_t>& cur_shape,
                                  const nlohmann::json& compile_info) {
    std::string tiling_id;
    if (!compile_info.contains(kCompileRepoSeeds) || !compile_info.contains(kCompileRepoRange)) {
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "no repo_sends or repo_range in compile info json");
      return tiling_id;
    }

    auto& tiling_seeds = compile_info.at(kCompileRepoSeeds);
    auto& repo_range = compile_info.at(kCompileRepoRange);
    int64_t min_dist = std::numeric_limits<int64_t>::max();
    for (auto it = tiling_seeds.begin(); it != tiling_seeds.end(); it++) {
      std::vector<int64_t> seed = it.value().get<std::vector<int64_t>>();
      auto& range = repo_range[it.key()];
      if (is_shape_in_range(cur_shape, range)) {
        int64_t dist = calc_dist(cur_shape, seed);
        if (dist < min_dist) {
          tiling_id = it.key();
          min_dist = dist;
        }
      }
    }

    if (tiling_id.empty()) {
      if (!compile_info.contains(kCompileCostRange)) {
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "no cost_range in compile info json");
        return tiling_id;
      }

      auto& costRange = compile_info.at(kCompileCostRange);
      for (auto it = costRange.begin(); it != costRange.end(); it++) {
        auto& range = it.value();
        if (is_shape_in_range(cur_shape, range)) {
          tiling_id = it.key();
          break;
        }
      }
    }

    return tiling_id;
  }

  void update_run_info(const std::vector<bool>& dynamic_mode,
                       const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& output_shape,
                       optiling::utils::OpRunInfo& run_info) {
    for (int32_t i = 0; i < kConv3dVarDimSizeLimit; ++i) {
      if (dynamic_mode[i]) {
        int32_t shape_index = kConv3DDynamicShapeDims[i];
        run_info.AddTilingData(static_cast<int32_t>(input_shape[shape_index]));
        if (i != 0) {
          run_info.AddTilingData(static_cast<int32_t>(output_shape[shape_index]));
        }
      }
    }
  }

  void update_run_info_cube(const std::vector<int64_t>& var_value, optiling::utils::OpRunInfo& run_info) {
    for (size_t i = 0; i < var_value.size(); ++i) {
      run_info.AddTilingData(static_cast<int32_t>(var_value[i]));
    }
  }

  string cube_tiling_batch(const std::string& op_type, const std::vector<int64_t>& cur_shape,
                         const nlohmann::json& compile_info, string tiling_id) {
    if (cur_shape.empty()) {
      return tiling_id;
    }

    if (!compile_info.contains("tiling_range")) {
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "no tiling_range in compile info json");
      return tiling_id;
    }
    auto& tiling_range = compile_info.at("tiling_range");
    for (auto it = tiling_range.begin(); it != tiling_range.end(); it++) {
      auto& range = it.value();
      if (is_shape_in_range_cube(cur_shape, range)){
        tiling_id = it.key();
      }
    }
    return tiling_id;
  }

  string cube_tiling_nhw(const std::string& op_type, const std::vector<int64_t>& cur_shape,
                       const nlohmann::json& compile_info, string tiling_id) {
    if (!compile_info.contains(kCompileRepoSeeds) || !compile_info.contains(kCompileRepoRange)) {
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "no repo_seeds or repo_range in compile info json");
        return tiling_id;
    }

    int32_t seedHDim = 1;
    int32_t seedWDim = 2;
    int32_t hDim = 2;
    int32_t wDim = 3;

    auto& repo_range = compile_info.at(kCompileRepoRange);
    auto& tiling_seeds = compile_info.at(kCompileRepoSeeds);
    int64_t min_dist = std::numeric_limits<int64_t>::max();
    for (auto it = tiling_seeds.begin(); it != tiling_seeds.end(); it++) {
      std::vector<int32_t> seed = it.value().get<std::vector<int32_t>>();
      auto& range = repo_range[it.key()];

      if (is_shape_in_range_cube(cur_shape, range)) {
        int32_t dist = abs(cur_shape[hDim] - seed[seedHDim]) + abs(cur_shape[wDim] - seed[seedWDim]);
        if (dist < min_dist) {
          tiling_id = it.key();
          min_dist = dist;
        }
      }
    }
    if (tiling_id.empty()) {
      if (!compile_info.contains(kCompileCostRange)) {
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "no cost_range in compile info json");
        return tiling_id;
      }

      auto& cost_range = compile_info.at(kCompileCostRange);
      for (auto it = cost_range.begin(); it != cost_range.end(); it++) {
        auto& range = it.value();
        if (is_shape_in_range_cube(cur_shape, range)) {
          tiling_id = it.key();
          break;
        }
      }
    }
    return tiling_id;
  }

  void deal_with_compile_info_array(const nlohmann::json& compile_info, nlohmann::json &opInfo) {
    nlohmann::json item;
    opInfo = compile_info[0];
    for (size_t i = 1; i < compile_info.size(); i++) {
      item = compile_info[i];
      std::vector<std::string> key_list = {kCompileRepoSeeds, kCompileRepoRange, kCompileCostRange};
      for (auto key : key_list) {
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
  }
}

namespace optiling {
 /*
  * @brief: tiling function of conv2d forward and backprop
  * @param [in] op_type: op_type of ops
  * @param [in] input_shape: input shape of ops
  * @param [in] var_map: variable name and value passed in at runtime
  * @param [in] compile_info: compilation information includes tiling coverage
  * @param [out] run_info: runtime information
  * @return bool: success or not
  */
  bool cube_tiling1(const std::string& op_type,
                  const std::vector<int64_t>& input_shape,
                  const std::vector<int64_t>& var_value,
                  const nlohmann::json& compile_info,
                  utils::OpRunInfo& run_info) {
    try {
      OP_LOGD(op_type.c_str(), "input compile info: %s", compile_info.dump().c_str());
      std::vector<std::string> vars = compile_info.at("_vars").begin().value().get<std::vector<std::string>>();
      std::string tiling_id("");

      if (compile_info["tiling_type"] == "default_tiling") {
        std::vector<int64_t> default_range = compile_info["default_range"].begin().value().get<std::vector<int64_t>>();
        if (is_shape_in_range_cube(input_shape, default_range)) {
          tiling_id = compile_info["default_range"].begin().key();
        }
      } else if (vars.size() != 1) {
        tiling_id = cube_tiling_nhw(op_type, input_shape, compile_info, tiling_id);
      } else {
        tiling_id = cube_tiling_batch(op_type, input_shape, compile_info, tiling_id);
      }

      if (tiling_id.empty()) {
          if (compile_info.contains("correct_range_flag") && compile_info["correct_range_flag"]) {
              CUBE_INNER_ERR_REPORT(op_type.c_str(), "The original range does not meet requirements,"
                "new range is generated during op compile, but the shape is not covered by new range");
          }
          CUBE_INNER_ERR_REPORT(op_type.c_str(), "This shape is not covered by any tiling,"
                                                 "please modify range and recompile");
          return false;
      }

      if (!compile_info.contains("block_dim")) {
          CUBE_INNER_ERR_REPORT(op_type.c_str(), "no block_dim in compile info json");
          return false;
      }

      OP_LOGD(op_type.c_str(), "get tiling_id: %s", tiling_id.c_str());
      run_info.SetBlockDim(static_cast<uint32_t>(compile_info["block_dim"][tiling_id]));
      run_info.SetTilingKey(std::stoi(tiling_id));

      update_run_info_cube(var_value, run_info);
      return true;
    } catch (...) {
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "get unknown exception, please check compile info json.");
      return false;
    }
  }

  /*
   * @brief: tiling function of cube operators
   * @param [in] compile_info: compile time generated info of operator
   * @param [in] opInfo: merge compile time generated info of operator
   * @param [out] runInfo: result data
   * @return : void
   */
  void deal_with_compile_info(const nlohmann::json& compile_info,
                              nlohmann::json &opInfo)
  {
    if (compile_info.is_object()) {
      opInfo = compile_info;
    } else if (compile_info.is_array()) {
      deal_with_compile_info_array(compile_info, opInfo);
    }
  }

  /*
  * @brief: tiling function of conv3d forward and backprop
  * @param [in] op_type: op_type of the conv3d forward and backprop
  * @param [in] input_shape: input shape of the conv3d forward and backprop
  * @param [in] output_shape: output shape of the conv3d forward and backprop
  * @param [in] compile_info: compile time generated info of the conv3d forward and backprop
  * @param [out] run_info: result data
  * @return bool: success or not
  */
  bool Conv3DCommonTiling(const std::string& op_type,
                          const std::vector<int64_t>& input_shape,
                          const std::vector<int64_t>& output_shape,
                          const nlohmann::json& compile_info,
                          utils::OpRunInfo& run_info) {
    std::vector<std::string> all_vars;
    get_var_names(op_type, all_vars);
    if (all_vars.size() != kConv3dVarDimSizeLimit) {
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "found unrecoginzed var name list.");
      return false;
    }

    try {
      OP_LOGD(op_type.c_str(), "input compile info: %s", compile_info.dump().c_str());
      std::vector<bool> dynamic_batch = {true, false, false, false};
      std::vector<std::string> vars = *(compile_info.at("_vars").begin());
      std::vector<bool> dynamic_mode(kConv3dVarDimSizeLimit, false);
      for (int32_t i = 0; i < kConv3dVarDimSizeLimit; ++i) {
        dynamic_mode[i] = std::find(vars.begin(), vars.end(), all_vars[i]) != vars.end();
      }

      std::string tiling_id;
      bool is_dynamic_batch = (dynamic_batch == dynamic_mode);
      if (compile_info["tiling_type"] == "default_tiling") {
        std::vector<int64_t> default_range = compile_info["default_range"].begin().value().get<std::vector<int64_t>>();
        if (is_shape_in_range(input_shape, default_range)) {
          tiling_id = compile_info["default_range"].begin().key();
        }
      } else if (is_dynamic_batch) {
        tiling_id = get_conv3D_batch_tiling(op_type, input_shape, compile_info);
      } else {
        tiling_id = get_conv3D_ndhw_tiling(op_type, input_shape, compile_info);
      }

      if (tiling_id.empty()) {
        if (op_type == "Conv3D" && compile_info["correct_range_flag"]) {
          CUBE_INNER_ERR_REPORT(op_type.c_str(), "The original range does not meet requirements,"
                              "new range is generated during op compile, but the shape is not covered by new range");
        }
        CUBE_INNER_ERR_REPORT(op_type.c_str(), 
            "This shape is not covered by any tiling, please modify range and recompile");
        return false;
      }

      if (!compile_info.contains("block_dim")) {
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "no block_dim in compile info json");
        return false;
      }

      OP_LOGD(op_type.c_str(), "get tiling_id: %s", tiling_id.c_str());
      run_info.SetBlockDim(static_cast<uint32_t>(compile_info["block_dim"][tiling_id]));
      run_info.SetTilingKey(std::stoi(tiling_id));

      update_run_info(dynamic_mode, input_shape, output_shape, run_info);
      return true;
    } catch (...) {
      CUBE_INNER_ERR_REPORT(op_type.c_str(), "get unknown exception, please check compile info json.");
      return false;
    }
  }
}  // namespace optiling