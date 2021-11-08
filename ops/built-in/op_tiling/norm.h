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
 * \file norm.h
 * \brief
 */

#ifndef NORM_TILING_H
#define NORM_TILING_H

#include <cmath>
#include <vector>

#include "vector_tiling.h"

namespace optiling {

struct TilingInfoNorm {
  int32_t block_dim{-1};
  int32_t block_tiling_axis{-1};
  int64_t block_tiling_factor{-1};
  int32_t ub_tiling_axis{-1};
  int64_t ub_tiling_factor{-1};
};

struct ReorderInfoNorm {
  std::vector<int64_t> reorder_input_shape{std::vector<int64_t>(10, 0)};
  std::vector<int32_t> fused_block_tiling_axis;
  // pos after reorder : pos before reorder
  //    vector.idx     :      vector[idx]
  std::vector<int32_t> reorderPos_oriPos{std::vector<int32_t>(10, 0)};
};

struct CompileInfoNorm {
  bool is_const{false};
  bool is_const_post{false};
  bool is_keep_dims{false};
  bool is_fuse_axis{true};
  int32_t max_ub_count{-1};
  int32_t workspace_max_ub_count{-1};
  int32_t pad_max_ub_count{-1};
  int32_t pad_max_entire_size{-1};
  int32_t core_num{-1};
  int32_t min_block_size{-1};
};

class Norm {
  public:
    explicit Norm(const std::string& _op_type, const ge::Operator& _op_paras, const nlohmann::json& _op_info,
                  utils::OpRunInfo& _run_info)
        : op_type(_op_type), op_paras(_op_paras), op_info(_op_info), run_info(_run_info) {
    }
    ~Norm() {
    }
    bool GetInput();
    bool Init();
    bool FusedReduceAxis();
    bool GetCompileInfo();
    bool ProcessTiling();
    bool GetWorkspaceBlockTilingInfo();
    bool GetBlockTilingInfo();
    bool ProcessReorderAxis();
    bool PartialReorderUbTiling();
    bool GetUbTilingInfo();
    bool NeedRefineBlockTiling();
    bool DoTiling();
    bool ConstInputProcPost();
    bool CalcTilingKey();
    bool CalcWorkspace();
    bool WriteTilingData();

  private:
    bool IsInVector(std::vector<int32_t>& shape, int32_t value);
    int64_t CalcAfterReduceShapeProduct(std::vector<int64_t>& shape, std::vector<int32_t>& axis);
    int64_t CalcReduceShapeProduct(std::vector<int64_t>& shape, std::vector<int32_t>& axis);
    int32_t CalcPattern(std::vector<int64_t>& shape, std::vector<int32_t>& axis);
    bool IsNeedWorkspace();
    bool GetVarValue();
    int32_t GetBlockDim(int32_t tiling_axis, int64_t tiling_factor);
    int64_t CalcReorderShapeProduct(int32_t axis_index, int32_t block_tiling_axis_in_reorder);
    int64_t CalcReorderShapeProductAlign(int32_t axis_index, int32_t block_tiling_axis_in_reorder);

  private:
    const std::string& op_type;
    const ge::Operator& op_paras;
    const nlohmann::json& op_info;
    utils::OpRunInfo& run_info;
    CompileInfoNorm compileInfo;
    TilingInfoNorm tilingInfo;
    ReorderInfoNorm reorderInfo;

    std::vector<int64_t> input_shape_ori{std::vector<int64_t>(10, 0)};
    std::vector<int32_t> reduce_axis_ori{std::vector<int32_t>(10, 0)};
    std::vector<int64_t> input_shape{std::vector<int64_t>(10, 0)};
    std::vector<int32_t> reduce_axis{std::vector<int32_t>(10, 0)};

    // assistant
    std::vector<int64_t> input_align_shape{std::vector<int64_t>(10, 0)};
    std::vector<int32_t> reduce_flag{std::vector<int32_t>(10, 0)};
    std::vector<int64_t> workspace{std::vector<int64_t>(10, 0)};
    std::vector<int32_t> var_value{std::vector<int32_t>(10, 0)};

    bool is_last_axis_reduce{false};
    bool is_need_workspace{false};
    bool is_partial_reorder{false};
    bool is_split_block{true};
    bool is_align_and_remove_pad{false};
    int64_t shape_after_reduce_product{-1};
    int64_t reduce_product{-1};
    int32_t last_r_axis_index{-1};
    int32_t first_a_axis_index{-1};

    int32_t pattern{-1};
    int32_t sch_type{0};
    int32_t db{0};
    int32_t block_size{-1};
    int64_t ub_size{0};
    int32_t tiling_key{-1};
};

/*
 * @brief: tiling function of norm operator
 * @param [in] op_type: op_type of the norm operator
 * @param [in] op_paras: inputs/outputs/attrs of the norm operator
 * @param [in] op_compile_info: compile time generated info of the norm operator
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool NormTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                utils::OpRunInfo& run_info);

class NormCompileInfo: public AutoTilingCompileInfo {
  public:
  NormCompileInfo(const std::string& o, const std::string& p, const nlohmann::json& c)
    : AutoTilingCompileInfo(o, p), compile_info(c) {}
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const override;
  bool DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info, const OpInfo& op_info) const override;

  private:
  const nlohmann::json compile_info;
};

std::shared_ptr<AutoTilingCompileInfo> CreateNormTilingHandler(const std::string& op_type,
                                                               const std::string& pattern,
                                                               const nlohmann::json& parsed_compile_info);

}  // namespace optiling

#endif  // NORM_TILING_H
