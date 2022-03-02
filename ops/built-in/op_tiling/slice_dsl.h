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
 * \file slice_dsl.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_SLICE_SCHEDULE_H_
#define OPS_BUILT_IN_OP_TILING_SLICE_SCHEDULE_H_

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

#include "vector_tiling.h"
#include "external/graph/operator.h"

namespace optiling {
  constexpr size_t SLICE_INIT_DIM_LEN = 8;
  constexpr int64_t DEFAULT_ALIGN_VALUE = 32;
  struct SliceDslCompileInfo {
    // construct func
    SliceDslCompileInfo() = default;
    SliceDslCompileInfo(const std::string& _op_type, const nlohmann::json &compile_info);

    // base info
    int64_t core_num{0};
    int64_t ub_size{1};
    int64_t x_type_size{1};
    int64_t x_align{DEFAULT_ALIGN_VALUE};
    int64_t tensor_nums{1};
    int64_t end_mode{0};
    bool is_const_begins{false};
    bool is_const_ends{false};
    bool is_const_sizes{false};
    bool is_static{false};
    bool is_const{false};
    int64_t const_key{0};
    int64_t const_block_dims{0};

    std::vector <int64_t> size{};
    std::vector <int64_t> begin{};
    std::vector <int64_t> end{};
    std::vector <int64_t> coex_list{};

    std::unordered_map<std::string, std::vector<int32_t>> slice_vars;

  };

  class SliceDsl {
   public:
    explicit SliceDsl(const std::string& _op_type, const ge::Operator& _op_paras,
                      const SliceDslCompileInfo& _slice_compile_info, utils::OpRunInfo& _run_info):
          op_type(_op_type), op_paras(_op_paras), slice_compile_info(_slice_compile_info), run_info(_run_info) {
    }

    ~SliceDsl() = default;
    bool DoTiling();

   private:
    bool Init();
    void SimplyShape(std::vector<int64_t> org_x_shape, std::vector<int64_t> org_begin_list,
                     std::vector<int64_t> org_size_list);

    bool DoBaseTiling();

    bool IsBothAlignTiling();

    bool IsDePadTiling(int64_t last_dim_align);

    bool IsLRDePadTiling();

    bool IsStrideAlignTiling();

    bool DoBlockUbTiling();

    bool DoBlockTiling();

    bool DoUbTiling();

    bool TryImproveTiling();

    bool DoOneDimTiling();

    bool CalcKey();

    bool WriteTilingData();

    const std::string &op_type;
    const ge::Operator &op_paras;
    const SliceDslCompileInfo& slice_compile_info;
    utils::OpRunInfo& run_info;
    std::vector <int64_t> input_x_shape{std::vector<int64_t>(SLICE_INIT_DIM_LEN, 0)};
    std::vector <int64_t> x_shape{std::vector<int64_t>(SLICE_INIT_DIM_LEN, 0)};
    std::vector <int64_t> begin_list{std::vector<int64_t>(SLICE_INIT_DIM_LEN, 0)};
    std::vector <int64_t> size_list{std::vector<int64_t>(SLICE_INIT_DIM_LEN, 0)};

    size_t shape_len{1};
    int64_t pre_dim{1};
    int64_t last_dim{1};
    int64_t last_dim_align{1};
    int64_t x_last_dim_align{1};
    int64_t ub_available{1};
    int64_t mode{0};
    int64_t total_size{1};
    bool is_zero_shape{false};

    size_t block_axis{0};
    size_t ub_axis{0};
    int64_t block_dims{1};
    int64_t block_factor{1};
    int64_t ub_factor{1};

    int64_t key{-1};
  };

  class SliceTilingHandler: public AutoTilingHandler {
   public:
    SliceTilingHandler(const std::string &o, const std::string &p, const nlohmann::json &c)
        : AutoTilingHandler(o, p), slice_compile_info(o, c) {}

    ~SliceTilingHandler() = default;

    bool DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info) const override;
    bool DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info, const OpInfo &op_info) const override;

   private:
    const SliceDslCompileInfo slice_compile_info;
  };
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_SLICE_SCHEDULE_H_