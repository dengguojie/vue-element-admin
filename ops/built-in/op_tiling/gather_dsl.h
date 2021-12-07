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

#include "vector_tiling.h"
#include "external/graph/operator.h"

namespace optiling {
  struct GatherDslCompileInfo {
    // construct func
    GatherDslCompileInfo() = default;
    GatherDslCompileInfo(const std::string& _op_type, const nlohmann::json &compile_info);

    // base info
    int64_t core_num{0};
    int64_t ub_size{1};
    int64_t l1_size{0};
    int64_t gather_type{0};
    int64_t params_dtype{0};
    int64_t params_align{32};
    int64_t indices_dtype{0};

    // custom info
    int64_t params_l1_num{1};
    int64_t params_ub_half_num{1};
    int64_t batch_dims{0};
    bool is_binary_shape{false};
    int64_t org_batch_dims{0};

    // tensor size
    std::unordered_map<std::string, std::vector<int64_t>> tensor_sizes;

    // gather vars
    std::unordered_map<std::string, std::vector<int32_t>> gather_vars;

    // const axis info
    bool is_dynamic_const{false};
    int32_t const_axis{0};
  };

  class GatherDsl {
    public:
      explicit GatherDsl(const std::string& _op_type, const ge::Operator& _op_paras,
                                 const GatherDslCompileInfo& _gather_compile_info, utils::OpRunInfo& _run_info) :
              op_type(_op_type), op_paras(_op_paras), gather_compile_info(_gather_compile_info), run_info(_run_info){
      }

      ~GatherDsl() = default;

      bool DoTiling();

    private:
      bool Init();

      void SimplyParamsAndIndices(std::vector <int64_t> org_params_shape,
                                  std::vector <int64_t> org_indices_shape);
      bool IsZeroShapeTiling();
      bool DoZeroShapeTiling();

      bool IsSpecialPattern();

      bool IsDePadTiling();
      bool DoDePadTiling();

      bool IsScalarTiling();
      bool DoScalarTiling();

      bool IsStoreUB(int64_t params_total);
      bool IsStoreL1(int64_t params_total);

      bool IsParamsUbTiling();
      bool DoParamsUbTiling();

      bool IsParamsL1Tiling();
      bool DoParamsL1Tiling();

      bool IsDbModule();
      bool DoDbModule();

      bool IsBaseTiling();

      void BlockFirstAxis();
      void BlockSecondAxis();
      void BlockThirdAxis();
      void BlockLastAxis();
      void EnsureBlockUBTiling();

      bool DoBaseTiling();

      bool CalcKey();

      bool WriteTilingData();

      const std::string &op_type;
      const ge::Operator &op_paras;
      const GatherDslCompileInfo &gather_compile_info;
      utils::OpRunInfo &run_info;

      std::vector <int64_t> params_shape{};
      std::vector <int64_t> indices_shape{};
      std::vector <int64_t> output_shape{};

      int64_t rank{1};
      int64_t axis{0};
      int64_t params_rows{1};
      int64_t real_batch_dims{0};

      int64_t params_size_total{1};
      int64_t indices_size_total{1};
      int64_t total_size{1};

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

  class GatherTilingHandler : public AutoTilingHandler {
  public:
    GatherTilingHandler(const std::string &o, const std::string &p, const nlohmann::json &c)
            : AutoTilingHandler(o, p), gather_compile_info(o, c) {}

    ~GatherTilingHandler() {}

    bool DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info) const override;
    bool DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info, const OpInfo &op_info) const override;

  private:
    const GatherDslCompileInfo gather_compile_info;
  };

  std::shared_ptr <AutoTilingHandler> CreateGatherTilingHandler(const std::string &op_type,
                                                               const std::string &pattern,
                                                               const nlohmann::json &parsed_compile_info);
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_GATHER_SCHEDULE_H_

