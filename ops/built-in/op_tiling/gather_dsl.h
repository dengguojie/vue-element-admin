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
 * \file gather_schedule.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_GATHER_SCHEDULE_H_
#define OPS_BUILT_IN_OP_TILING_GATHER_SCHEDULE_H_

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_run_context.h"

#include "vector_tiling.h"
#include "auto_tiling.h"
#include "auto_tiling_context.h"

namespace optiling {
  constexpr std::size_t GATHER_INIT_DIM_LEN = 11;
  constexpr std::size_t INDICES_INIT_DIM_LEN = 3;
  constexpr std::size_t OUTPUT_INIT_DIM_LEN = 4;
  struct GatherDslCompileInfo : AutoTilingCompileInfo {
    // construct func
    GatherDslCompileInfo() = default;
    GatherDslCompileInfo(const std::string& _op_type, const nlohmann::json &compile_info);
    ~GatherDslCompileInfo() override = default;
    bool Parse(const char* op_type, const nlohmann::json &compile_info);

    // base info
    int64_t core_num{0};
    int64_t ub_size{1};
    int64_t gather_type{0};
    int64_t params_dtype{0};
    int64_t params_align{32};
    int64_t indices_dtype{0};

    // custom info
    int64_t params_ub_store_num{1};
    int64_t batch_dims{0};
    bool unknown_batch_dims{false};
    int64_t org_batch_dims{0};
    string attr_name;
    size_t attr_idx{0};

    // tensor size
    std::unordered_map<std::string, std::vector<int64_t>> tensor_sizes;

    // gather vars
    std::unordered_map<std::string, std::vector<int32_t>> gather_vars;

    // const axis info
    bool is_dynamic_const{false};
    int32_t const_axis{0};
    bool is_valid{false};
  };

  template <typename T>
  class GatherDsl {
    public:
      explicit GatherDsl(T* _context, const OpInfoImpl* _op_info)
          : context(_context), op_info(_op_info) {}

      ~GatherDsl() = default;

      bool DoTiling();

    private:
      bool Init();
      void GetRealBatchDims();
      void SimplyParamsAndIndices();
      bool IsZeroShapeTiling();
      bool DoZeroShapeTiling();

      bool IsSpecialPattern();

      bool IsDePadTiling();
      bool DoDePadTiling();

      bool IsScalarTiling();
      bool DoScalarTiling();

      bool IsStoreUB(int64_t params_total);

      bool IsParamsUbTiling();
      bool DoParamsUbTiling();

      bool IsDbModule();
      bool DoDbModule();

      bool IsBaseTiling();

      void BlockFirstAxis();
      void BlockSecondAxis();
      void BlockThirdAxis();
      void BlockLastAxis();
      void EnsureBlockUBTiling();
      void SafeTiling();
      bool DoBaseTiling();

      bool CalcKey();

      bool WriteTilingData();

      T* context;
      const OpInfoImpl* op_info{nullptr};
      const char* op_type;
      const GatherDslCompileInfo* gather_compile_info;

      std::array <int64_t, GATHER_INIT_DIM_LEN> org_params_shape{};
      std::array <int64_t, GATHER_INIT_DIM_LEN> org_indices_shape{};
      size_t cur_params_dim_len{0};
      size_t cur_indices_dim_len{0};

      std::vector <int64_t> params_shape{std::vector<int64_t>(GATHER_INIT_DIM_LEN, 0)};
      std::vector <int64_t> indices_shape{std::vector<int64_t>(INDICES_INIT_DIM_LEN, 0)};
      std::vector <int64_t> output_shape{std::vector<int64_t>(OUTPUT_INIT_DIM_LEN, 0)};

      int64_t rank{1};
      int64_t axis{0};
      int64_t params_rows{1};
      int64_t params_rows_align{1};
      size_t real_batch_dims{0};

      int64_t params_size_total{1};
      int64_t indices_size_total{1};
      int64_t total_size{1};
      int64_t output_size{1};

      int64_t key{-1};

      size_t block_axis{0};
      size_t ub_axis{0};
      int64_t block_dims{1};
      int64_t block_factor{1};
      int64_t ub_factor{1};
      int32_t key_special_pattern{0};

      int64_t params_num_ub{0};
      int64_t indices_num_ub{0};
      int64_t real_params_row_num{0};
  };
  template class GatherDsl<AutoTilingContext>;
  template class GatherDsl<AutoTilingOp>;

  class GatherTilingHandler : public AutoTilingHandler {
  public:
    GatherTilingHandler(const std::string &o, const std::string &p, const nlohmann::json &c)
        : AutoTilingHandler(o, p), gather_compile_info(o, c) {}

    ~GatherTilingHandler() = default;

    bool DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info) const override;
    bool DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info, const OpInfo &op_info) const override;

  private:
    const GatherDslCompileInfo gather_compile_info;
  };
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_GATHER_SCHEDULE_H_
