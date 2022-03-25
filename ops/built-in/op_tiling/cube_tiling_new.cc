/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
#include <cstdlib>
#include <string>

#include "../op_proto/util/error_util.h"
#include "cube_tiling_new.h"

namespace {
  constexpr int32_t kConv3dDimSizeLimit = 6;
  constexpr int32_t kConv3dVarDimSizeLimit = 4;
  static const int kRangeDimLen = 2;
  static const int kBlockSize = 16;
  static const int32_t kHoshWNoDivided = 2;
  static const int kNumTwo = 2;
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


  static bool is_shape_in_range_cube(const std::vector<int64_t> &shape, const std::vector<int64_t> &range) {
    const std::vector<int32_t> shape_dim = {0, 2, 3};
    const std::vector<int32_t> range_dim = {0, 1, 2, 3, 4, 5};
    if (range.size() == range_dim.size()) {
      for (size_t i = 0; i < shape_dim.size(); ++i) {
        if (shape[shape_dim[i]] < range[range_dim[i * kRangeDimLen]] ||
            shape[shape_dim[i]] > range[range_dim[i * kRangeDimLen + 1]]) {
          return false;
        }
      }
    } else if (range.size() == kRangeDimLen) {
      if (shape[shape_dim[0]] < range[0] || shape[shape_dim[0]] > range[1]) {
        return false;
      }
    } else {
      return false;
    }

    return true;
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
      if (is_shape_in_range_cube(cur_shape, range)) {
        tiling_id = it.key();
      }
    }
    return tiling_id;
  }

  string cube_tiling_nhw(const std::string& op_type, const std::vector<int64_t>& cur_shape,
                         const nlohmann::json& compile_info, string tiling_id) {
    if (!compile_info.contains("repo_seeds") || !compile_info.contains("repo_range")) {
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "no repo_seeds or repo_range in compile info json");
        return tiling_id;
    }

    int32_t seedHDim = 1;
    int32_t seedWDim = 2;
    int32_t hDim = 2;
    int32_t wDim = 3;

    const auto& repo_range = compile_info.at("repo_range");
    auto& tiling_seeds = compile_info.at("repo_seeds");
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
      if (!compile_info.contains("cost_range")) {
        CUBE_INNER_ERR_REPORT(op_type.c_str(), "no cost_range in compile info json");
        return tiling_id;
      }

      auto& cost_range = compile_info.at("cost_range");
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
  inline int32_t Lcm(const int32_t &param1, const int32_t &param2) {
    int32_t pram1_lcm = param1;
    int32_t pram2_lcm = param2;
    int32_t temp = pram1_lcm * pram2_lcm;
    int32_t param1_temp = pram1_lcm;
    while (pram1_lcm % pram2_lcm != 0) {
      param1_temp = pram1_lcm;
      pram1_lcm = pram2_lcm;
      pram2_lcm = param1_temp % pram2_lcm;
    }
    return temp / pram2_lcm;
  }

  void SetRunInfoStrideOne(const RunInfoRaras &run_info_params, const DxParas &params, const Tiling &tiling,
                           utils::OpRunInfo &run_info, RunInfoParaStrideEqualOne &run) {
    run.batch = params.batch;
    run.co = params.co;
    run.ho = params.ho;
    run.wo = params.wo;
    run.filter_cin1hw = params.filter_cin1hw;
    run.filter_cout1 = params.filter_cout1;
    run.cin = params.cin;
    run.c1 = params.c1;
    run.h = params.h;
    run.w = params.w;
    run.kh = params.kh;
    run.kw = params.kw;
    run.g_extend = run_info_params.g_extend;
    run.dx_c1_extend = run_info_params.dx_c1_extend;
    run.multiple_extend = run_info_params.multiple_extend;
    run.padu = params.padu;
    run.padd = params.padd;
    run.padl = params.padl;
    run.padr = params.padr;
    run.shape_up_modify = run_info_params.shape_up_modify;
    run.shape_left_modify = run_info_params.shape_left_modify;
    run.shape_down_modify = run_info_params.shape_down_modify;
    run.shape_right_modify = run_info_params.shape_right_modify;
    run.pad_up_before = run_info_params.pad_up_before;
    run.pad_left_before = run_info_params.pad_left_before;
    run.pad_down_after = run_info_params.pad_down_after;
    run.pad_right_after = run_info_params.pad_right_after;
    run.batch_dim = tiling.batch_dim;
    run.n_dim = tiling.n_dim;
    run.m_dim = tiling.m_dim;
    run.batch_single_core = run_info_params.batch_single_core;
    run.m_al1 = tiling.m_al1;
    run.n_bl1 = tiling.n_bl1;
    run.k_aub = tiling.k_aub * params.kh * params.kw * kBlockSize;
    run.m_aub = tiling.m_aub;
    run.m_l0 = tiling.m_l0;
    run.n_l0_div_ub = run_info_params.n_l0_div_ub;
    run.n_cub = tiling.n_cub;
    run.k_l0 = tiling.k_l0;
    run.min_kl1_div_kl0 = run_info_params.min_kl1_div_kl0;
    run.max_kl1_div_min_kl1 = run_info_params.max_kl1_div_min_kl1;
    run.k_div_max_kl1 = run_info_params.k_div_max_kl1;
    run.al1_bound = run_info_params.al1_bound;
    run.bl1_bound = run_info_params.bl1_bound;
    run.aub_bound = run_info_params.aub_bound;
    run_info.AddTilingData(run);
  }

 void SetRunInfoStrideLargeOne(const RunInfoRaras &run_info_params, const DxParas &params, const Tiling &tiling,
                               utils::OpRunInfo &run_info, RunInfoParaStrideLargeOne &run) {
    run.filter_cin1hw = params.filter_cin1hw;
    run.filter_cout1 = params.filter_cout1;
    run.batch = params.batch;
    run.co1 = params.co1;
    run.ho = params.ho;
    run.wo = params.wo;
    run.cin = params.cin;
    run.c1 = params.c1;
    run.h = params.h;
    run.w = params.w;
    run.kh = params.kh;
    run.kw = params.kw;
    run.g_extend = run_info_params.g_extend;
    run.dx_c1_extend = run_info_params.dx_c1_extend;
    run.multiple_extend = run_info_params.multiple_extend;
    run.padu = params.padu;
    run.padd = params.padd;
    run.padl = params.padl;
    run.padr = params.padr;
    run.stride_h = params.stride_h;
    run.stride_w = params.stride_w;
    run.shape_up_modify = run_info_params.shape_up_modify;
    run.shape_left_modify = run_info_params.shape_left_modify;
    run.shape_down_modify = run_info_params.shape_down_modify;
    run.shape_right_modify = run_info_params.shape_right_modify;
    run.pad_up_before = run_info_params.pad_up_before;
    run.pad_left_before = run_info_params.pad_left_before;
    run.pad_down_after = run_info_params.pad_down_after;
    run.pad_right_after = run_info_params.pad_right_after;
    run.batch_dim = tiling.batch_dim;
    run.n_dim = tiling.n_dim;
    run.m_dim = tiling.m_dim;
    run.batch_single_core = run_info_params.batch_single_core;
    run.m_al1 = tiling.m_al1;
    run.n_bl1 = tiling.n_bl1;
    run.k_aub = tiling.k_aub * params.kw * params.kh * kBlockSize;
    run.m_aub = tiling.m_aub;
    run.m_l0 = tiling.m_l0;
    run.n_l0_div_ub = run_info_params.n_l0_div_ub;
    run.n_cub = tiling.n_cub;
    run.k_l0 = tiling.k_l0;
    run.min_kl1_div_kl0 = run_info_params.min_kl1_div_kl0;
    run.max_kl1_div_min_kl1 = run_info_params.max_kl1_div_min_kl1;
    run.k_div_max_kl1 = run_info_params.k_div_max_kl1;
    run.al1_bound = run_info_params.al1_bound;
    run.bl1_bound = run_info_params.bl1_bound;
    run.aub_bound = run_info_params.aub_bound;
    run_info.AddTilingData(run);
  }

  void SetRunInfo(const RunInfoRaras &run_info_params, const DxParas &params, const Tiling &tiling,
                  utils::OpRunInfo &run_info) {
    bool stride_equal_one = params.stride_h == 1 && params.stride_w == 1;
    if (stride_equal_one) {
      RunInfoParaStrideEqualOne run;
      SetRunInfoStrideOne(run_info_params, params, tiling, run_info, run);
    } else {
      RunInfoParaStrideLargeOne run;
      SetRunInfoStrideLargeOne(run_info_params, params, tiling, run_info, run);
    }
  }

  bool UpdateRunInfoBinary(const DxParas &params, const Tiling &tiling,
                           const int32_t &tiling_id, utils::OpRunInfo &run_info) {
    RunInfoRaras run_info_params;
    run_info_params.dy_c_ori = ((params.co + params.groups - 1) / params.groups) * params.groups;
    int32_t block_size = kBlockSize;
    int32_t dx_c_extend = Lcm(params.cin, block_size) / params.cin;
    int32_t dy_c_extend = Lcm(run_info_params.dy_c_ori, block_size) / run_info_params.dy_c_ori;
    run_info_params.multiple_extend = min(Lcm(dx_c_extend, dy_c_extend), params.groups);
    run_info_params.g_extend = (params.groups + run_info_params.multiple_extend - 1) / run_info_params.multiple_extend;
    run_info_params.dx_c1_extend = (run_info_params.multiple_extend * params.cin + kBlockSize - 1) / kBlockSize;
    run_info_params.pad_up_before = (params.kh - 1) * params.dilations_h - params.padu;
    run_info_params.pad_left_before = (params.kw - 1) * params.dilations_w - params.padl;
    run_info_params.pad_down_after =
        params.h - params.ho * params.stride_h - run_info_params.pad_up_before + (params.kh - 1) * params.dilations_h;
    run_info_params.pad_right_after =
        params.w - params.wo * params.stride_w -
        run_info_params.pad_left_before + (params.kw - 1) * params.dilations_w;
    run_info_params.shape_up_modify = (run_info_params.pad_up_before - abs(run_info_params.pad_up_before)) / kNumTwo;
    run_info_params.shape_left_modify =
        (run_info_params.pad_left_before - abs(run_info_params.pad_left_before)) / kNumTwo;
    run_info_params.shape_down_modify =
        (run_info_params.pad_down_after - abs(run_info_params.pad_down_after)) / kNumTwo;
    run_info_params.shape_right_modify =
        (run_info_params.pad_right_after - abs(run_info_params.pad_right_after)) / kNumTwo;
    run_info_params.pad_up_before = (run_info_params.pad_up_before + abs(run_info_params.pad_up_before)) / kNumTwo;
    run_info_params.pad_left_before =
        (run_info_params.pad_left_before + abs(run_info_params.pad_left_before)) / kNumTwo;
    run_info_params.pad_down_after = (run_info_params.pad_down_after + abs(run_info_params.pad_down_after)) / kNumTwo;
    run_info_params.pad_right_after =
        (run_info_params.pad_right_after + abs(run_info_params.pad_right_after)) / kNumTwo;
    run_info_params.batch_single_core = params.batch / tiling.batch_dim;
    int32_t n_single_size = tiling.n_bl1 * tiling.n_l0;
    int32_t m_single_size = tiling.m_al1 * tiling.m_l0;
    if (tiling.n_bl1 == 0) {
      n_single_size = tiling.n_single_core_size;
    }
    if (tiling.m_al1 == 0) {
      m_single_size = tiling.m_single_core_size;
    }
    run_info_params.n_l0_div_ub = tiling.n_l0 / tiling.n_cub;
    int32_t hosh = (params.kh - 1) + m_single_size * kBlockSize / params.w + kHoshWNoDivided;
    if (m_single_size * kBlockSize < params.w) {
      hosh = (params.kh - 1) + kHoshWNoDivided;
    } else if (m_single_size * kBlockSize % params.w == 0) {
      hosh = (params.kh - 1) + m_single_size * kBlockSize / params.w;
    }
    run_info_params.al1_bound =  tiling.k_al1 * params.wo * params.stride_w * kBlockSize * hosh;
    if (tiling.m_al1 == 0) {
      run_info_params.al1_bound = tiling.k_al1 * kBlockSize *
                  ((params.wo * params.stride_w * params.ho * params.stride_h + kBlockSize - 1) / kBlockSize) *
                  kBlockSize;
    }
    run_info_params.bl1_bound =
        tiling.k_bl1 * params.kh * params.kw * n_single_size * kBlockSize * kBlockSize;
    run_info_params.aub_bound = tiling.k_aub * tiling.m_aub * params.wo * kBlockSize * params.stride_w;
    bool stride_equal_one = params.stride_h == 1 && params.stride_w == 1;
    if (stride_equal_one) {
      run_info_params.aub_bound = tiling.k_aub * kBlockSize *
          ((tiling.m_aub * params.wo + params.kw - 1 + kBlockSize - 1) / kBlockSize) * kBlockSize;
    }
    CHECK_OP_FUNC(tiling.k_al1 == 0, return false, "k_al1 is zero");
    CHECK_OP_FUNC(tiling.k_bl1 == 0, return false, "k_bl1 is zero");
    int32_t max_kl1 = max(tiling.k_al1, tiling.k_bl1);
    int32_t min_kl1 = min(tiling.k_al1, tiling.k_bl1);
    run_info_params.min_kl1_div_kl0 = min_kl1 * params.kh * params.kw / tiling.k_l0;
    run_info_params.max_kl1_div_min_kl1 = max_kl1 / min_kl1;
    run_info_params.k_div_max_kl1 = params.co1 / max_kl1;

    SetRunInfo(run_info_params, params, tiling, run_info);
    run_info.SetBlockDim(static_cast<uint32_t>(tiling.batch_dim * tiling.n_dim * tiling.m_dim));
    run_info.SetTilingKey(static_cast<int64_t>(tiling_id));
    return true;
  }

  bool cube_tiling(const std::string& op_type,
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
}  // namespace optiling
