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
 * \file cube_tiling.cpp
 * \brief
 */
#include <stdlib.h>
#include <string>
#include "cube_tiling.h"

namespace {
  constexpr int32_t kConv3dDimSizeLimit = 6;
  constexpr int32_t kConv3dVarDimSizeLimit = 4;
  const std::vector<int32_t> kConv3DDynamicShapeDims = {0, 1, 3, 4}; // format: NDC1HWC0
  const std::vector<int32_t> kConv3DDynamicRangeDims = {0, 1, 2, 3}; // foramt: NDHW

  const std::vector<std::string> kConv3DVarNames = {"batch_n", "fmap_d", "fmap_h", "fmap_w"};
  const std::vector<std::string> kConv3DBpInputVarNames = {"batch_n", "dedy_d", "dedy_h", "dedy_w"};
  const std::vector<std::string> kConv3DTransposeVarNames = {"batch_n", "dedy_d", "dedy_h", "dedy_w"};

  void GetVarNames(const std::string &op_type, std::vector<std::string> &vars) {
    if (op_type == "Conv3D") {
      vars = kConv3DVarNames;
      return;
    }

    if (op_type == "Conv3DTranspose") {
      vars = kConv3DTransposeVarNames;
      return;
    }

    if (op_type == "Conv3DBackpropInput") {
      vars = kConv3DBpInputVarNames;
      return;
    }
  }

  static bool IsShapeInRange(const std::vector<int64_t> &shape, const std::vector<int64_t> &range) {
    // shape: NDHWC, range: NDHW
    if (shape.size() < kConv3dDimSizeLimit || range.size() < (kConv3dVarDimSizeLimit * 2)) {
      return false;
    }

    for (int32_t i = 0; i < kConv3dVarDimSizeLimit; ++i) {
      int32_t shape_index = kConv3DDynamicShapeDims[i];
      int32_t range_index = kConv3DDynamicRangeDims[i];
      if (shape[shape_index] < range[range_index * 2] || shape[shape_index] > range[(range_index * 2) + 1]) {
        return false;
      }
    }

    return true;
  }

  static int64_t CalcDist(const std::vector<int64_t> &shape, const std::vector<int64_t> &seeds) {
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

  std::string GetConv3DBatchTiling(const std::string& op_type, const std::vector<int64_t>& cur_shape,
                                   const nlohmann::json& compile_info) {
    std::string tiling_id("0");
    if (cur_shape.empty()) {
      return tiling_id;
    }

    if (!compile_info.contains("tiling_range")) {
      OP_LOGE(op_type.c_str(), "no tiling_range in compile info json");
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

  std::string GetConv3DNDHWTiling(const std::string& op_type, const std::vector<int64_t>& cur_shape,
                                  const nlohmann::json& compile_info) {
    std::string tiling_id("0");
    if (!compile_info.contains("repo_seeds") || !compile_info.contains("repo_range")) {
      OP_LOGE(op_type.c_str(), "no repo_sends or repo_range in compile info json");
      return tiling_id;
    }

    auto& tiling_seeds = compile_info.at("repo_seeds");
    auto& repo_range = compile_info.at("repo_range");
    int64_t min_dist = std::numeric_limits<int64_t>::max();
    for (auto it = tiling_seeds.begin(); it != tiling_seeds.end(); it++) {
      std::vector<int64_t> seed = it.value().get<std::vector<int64_t>>();
      auto& range = repo_range[it.key()];
      if (IsShapeInRange(cur_shape, range)) {
        int64_t dist = CalcDist(cur_shape, seed);
        if (dist < min_dist) {
          tiling_id = it.key();
          min_dist = dist;
        }
      }
    }

    if (tiling_id == "0") {
      if (!compile_info.contains("cost_range")) {
        OP_LOGE(op_type.c_str(), "no cost_range in compile info json");
        return tiling_id;
      }

      auto& costRange = compile_info.at("cost_range");
      for (auto it = costRange.begin(); it != costRange.end(); it++) {
        auto& range = it.value();
        if (IsShapeInRange(cur_shape, range)) {
          tiling_id = it.key();
          break;
        }
      }
    }

    return tiling_id;
  }

  void UpdateRunInfo(const std::vector<bool>& dynamic_mode,
                     const std::vector<int64_t>& input_shape,
                     const std::vector<int64_t>& output_shape,
                     optiling::OpRunInfo& run_info) {
    for (int32_t i = 0; i < kConv3dVarDimSizeLimit; ++i) {
      if (dynamic_mode[i]) {
        int32_t shape_index = kConv3DDynamicShapeDims[i];
        optiling::ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(input_shape[shape_index]));
        if (i != 0) {
          optiling::ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(output_shape[shape_index]));
        }
      }
    }
  }
}

namespace optiling {

int32_t g_nDim = 0;
int32_t g_hDim = 1;
int32_t g_wDim = 2;
int32_t g_nMinDim = 0;
int32_t g_nMaxDim = 1;
int32_t g_hMinDim = 2;
int32_t g_hMaxDim = 3;
int32_t g_wMinDim = 4;
int32_t g_wMaxDim = 5;

string CubeTilingBatch(const std::vector<int32_t>& curShape, const nlohmann::json& opInfo) {
  std::string tilingID("0");
  auto& tilingRange = opInfo.at("tiling_range");
  for (auto it = tilingRange.begin(); it != tilingRange.end(); it++) {
    auto& range = it.value();
    if (curShape[g_nDim] >= range[g_nMinDim] && curShape[g_nDim] <= range[g_nMaxDim]) {
      tilingID = it.key();
    }
  }
  return tilingID;
}

string CubeTilingCostModel(const std::vector<int32_t>& curShape, const nlohmann::json& opInfo) {
  std::string tilingID("0");
  auto& costRange = opInfo.at("cost_range");
  for (auto it = costRange.begin(); it != costRange.end(); it++) {
    auto& range = it.value();

    if (curShape[g_nDim] >= range[g_nMinDim] && curShape[g_nDim] <= range[g_nMaxDim] &&
        curShape[g_hDim] >= range[g_hMinDim] && curShape[g_hDim] <= range[g_hMaxDim] &&
        curShape[g_wDim] >= range[g_wMinDim] && curShape[g_wDim] <= range[g_wMaxDim]) {
      tilingID = it.key();
    }
  }
  return tilingID;
}

string CubeTilingNHW(const std::vector<int32_t>& curShape, const nlohmann::json& opInfo) {
  int32_t seedHDim = 0;
  int32_t seedWDim = 1;
  int32_t minDist = 1000000;

  std::string tilingID("0");
  auto& repoRange = opInfo.at("repo_range");
  auto& tilingSeeds = opInfo.at("repo_seeds");

  for (auto it = tilingSeeds.begin(); it != tilingSeeds.end(); it++) {
    std::vector<int32_t> seed = it.value().get<std::vector<int32_t>>();
    auto& range = repoRange[it.key()];
    if (curShape[g_nDim] >= range[g_nMinDim] && curShape[g_nDim] <= range[g_nMaxDim] &&
        curShape[g_hDim] >= range[g_hMinDim] && curShape[g_hDim] <= range[g_hMaxDim] &&
        curShape[g_wDim] >= range[g_wMinDim] && curShape[g_wDim] <= range[g_wMaxDim]) {
        int32_t dist = abs(curShape[g_hDim] - seed[seedHDim]) + abs(curShape[g_wDim] - seed[seedWDim]);
        if (dist < minDist) {
          tilingID = it.key();
          minDist = dist;
        }
    }
  }
  if (tilingID != "0") {
    return tilingID;
  }

  tilingID = CubeTilingCostModel(curShape, opInfo);
  return tilingID;
}


int32_t CubeTiling(const std::string& opType, const std::vector<int32_t>& curShape, const nlohmann::json& opInfo,
                   OpRunInfo& runInfo) {
    std::vector<std::string> varMap = opInfo.at("_vars")["10000"];
    std::string tilingID("0");

    if (opInfo["tiling_type"] == "default_tiling") {
        tilingID = opInfo["default_range"].begin().key();
    } else if (varMap.size() != 1) {
        tilingID = CubeTilingNHW(curShape, opInfo);
    } else {
        tilingID = CubeTilingBatch(curShape, opInfo);
    }
    if (tilingID == "0") {
        OP_LOGE(opType.c_str(),
                "This shape is not covered by any tiling, "
                "please modify range and recompile");
        return false;
    }
    runInfo.block_dim = (uint32_t)opInfo["block_dim"][tilingID];
    return std::stoi(tilingID);
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
                          OpRunInfo& run_info) {
    std::vector<std::string> all_vars;
    GetVarNames(op_type, all_vars);
    if (all_vars.size() != kConv3dVarDimSizeLimit) {
      OP_LOGE(op_type.c_str(), "found unrecoginzed var name list.");
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
      if (op_type == "Conv3D" && compile_info["tiling_type"] == "default_tiling") {
        tiling_id = compile_info["default_range"].begin().key();
      } else if (is_dynamic_batch) {
        tiling_id = GetConv3DBatchTiling(op_type, input_shape, compile_info);
      } else {
        tiling_id = GetConv3DNDHWTiling(op_type, input_shape, compile_info);
      }

      if (tiling_id == "0") {
        if (op_type == "Conv3D") {
          if (compile_info["correct_range_flag"]) {
            OP_LOGE(op_type.c_str(), "The original range does not meet requirements,"
                                "new range is generated during op compile, but the shape is not covered by new range");
          }
        }
        OP_LOGE(op_type.c_str(), "This shape is not covered by any tiling, please modify range and recompile");
        return false;
      }

      if (!compile_info.contains("block_dim")) {
        OP_LOGE(op_type.c_str(), "no block_dim in compile info json");
        return false;
      }

      OP_LOGD(op_type.c_str(), "get tiling_id: %s", tiling_id.c_str());
      run_info.block_dim = static_cast<uint32_t>(compile_info["block_dim"][tiling_id]);
      run_info.tiling_key = std::stoi(tiling_id);
      int status = compile_info["push_status"];
      if (status == 0) {
        ByteBufferPut(run_info.tiling_data, run_info.tiling_key);
      }

      UpdateRunInfo(dynamic_mode, input_shape, output_shape, run_info);
      return true;
    } catch (...) {
      OP_LOGE(op_type.c_str(), "get unknown exception, please check compile info json.");
      return false;
    }
  }
}  // namespace optiling
