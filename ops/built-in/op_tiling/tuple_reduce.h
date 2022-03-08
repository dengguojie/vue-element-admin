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
 * \file tuple_reduce.h
 * \brief
 */

#ifndef TUPLE_REDUCE_H
#define TUPLE_REDUCE_H
#include <cmath>
#include <vector>
#include "vector_tiling.h"
#include "graph/utils/op_desc_utils.h"

#define TUPLE_REDUCE_CEILING(x, y) (((x) + (y) - 1) / (y))
#define TUPLE_REDUCE_REFINE(x, y) TUPLE_REDUCE_CEILING(x, TUPLE_REDUCE_CEILING(x, y))
#define TUPLE_REDUCE_ALIGN(x, y) ((y) * TUPLE_REDUCE_CEILING(x, y))
namespace optiling {
// CONSTANTS
constexpr std::size_t TUPLE_REDUCE_MAX_SHAPE_LEN = 8;
constexpr std::size_t TUPLE_REDUCE_MAX_INPUT_NUMS = 70;
constexpr std::int64_t SINGLE_CORE_THRESHOLD = 1024;
constexpr std::int64_t MINIMUM_OVERLAP_COEFFICIENT = 32;
const std::vector<int32_t> INDEX = {0, 1, 2, 3, 4, 5, 6, 7};

namespace TupleReduce {

// TupleReduceCompileInfo
struct TupleReduceCompileInfo {
  public:
  TupleReduceCompileInfo() = default;
  TupleReduceCompileInfo(const std::string &op_info, const nlohmann::json &json_info);

  public:
  // COMMON INFORMATION
  std::vector<int32_t> common_info;
  size_t core_num_idx = 0;
  int32_t core_num{-1};
  size_t ub_size_idx = 1;
  int32_t ub_size{-1};
  size_t block_size_idx = 2;
  int32_t block_size{-1};
  size_t atomic_support_idx = 3;
  bool atomic_support{false};
  size_t dim_var_idx = 4;
  int32_t dim_var{-1};

  // AXIS INFORMATION
  std::vector<int32_t> reduce_axis;
  std::vector<int32_t> disable_fuse_axes;
  std::vector<int32_t> fused_broadcast_axis;
  std::vector<int32_t> fused_reduce_axis;
  std::vector<int32_t> fusible_code;

  // DYNAMIC INFORMATION
  bool is_const{false};
  bool runtime{false};

  // GRAPH INFORMATION
  std::vector<int32_t> graph_info;
  std::size_t inputs_num_idx = 0;
  std::uint32_t inputs_num{0};
  std::size_t buffer_count_idx = 1;
  std::int32_t buffer_count{0};
  std::size_t max_dtype_size_idx = 2;
  std::int32_t max_dtype_size{0};
  std::size_t min_dtype_size_idx = 3;
  std::int32_t min_dtype_size{0};
  std::size_t keep_dims_idx = 4;
  bool keep_dims{false};
  std::vector<int32_t> shapes_length;
  std::int32_t max_shape_len{0};
  std::int64_t each_buffer_size{0};

  // PARSE STATUS
  bool parsed_success{true};

  private:
  bool GetCommonInfo(const std::string &op_type, const nlohmann::json &json_info);
  bool GetAxisInfo(const std::string &op_type, const nlohmann::json &json_info);
  bool GetDynamicMode(const std::string &op_type, const nlohmann::json &json_info);
  bool GetGraphInfo(const std::string &op_type, const nlohmann::json &json_info);
  bool CheckParseStatus(const std::string &op_type) const;
}; // TupleReduceCompileInfo


// TupleReduceTilingInfo
struct TupleReduceTilingInfo {
  int32_t block_dim{-1};
  int32_t block_tiling_axis{-1};
  int64_t block_tiling_factor{-1};
  int32_t ub_tiling_axis{-1};
  int64_t ub_tiling_factor{-1};

  bool atomic{false};
  int64_t tiling_key{-1};
}; // TupleReduceTilingInfo


// TupleReduce
class TupleReduce {
  public:
  explicit TupleReduce(const std::string &_op_type, const ge::Operator &_op_paras,
                       const TupleReduceCompileInfo &parsed_compile_info, utils::OpRunInfo &_run_info)
      : op_type(_op_type), op_paras(_op_paras), compileInfo(parsed_compile_info), run_info(_run_info) {}
  ~TupleReduce() {}

  bool DoTiling();

  private:
  const std::string &op_type;
  const ge::Operator &op_paras;
  const TupleReduceCompileInfo &compileInfo;
  utils::OpRunInfo &run_info;
  TupleReduceTilingInfo tupleReduceTilingInfo;

  private:
  bool last_axis_reduce {false};
  std::int64_t reduce_pattern = 0;
  std::int64_t tmp_product;
  std::size_t block_axis_lb = 0;
  std::size_t block_axis_ub = 0;
  std::size_t core_num = 1;
  std::size_t ub_axis_lb = 0;
  std::size_t ub_axis_ub = 01;

  private:
  std::array<std::array<int64_t, TUPLE_REDUCE_MAX_SHAPE_LEN>, TUPLE_REDUCE_MAX_INPUT_NUMS> inputs_shape{};
  std::vector<int64_t> fused_shape{std::vector<int64_t>(TUPLE_REDUCE_MAX_SHAPE_LEN, 0)};
  std::vector<int32_t> reduce_one_hot{std::vector<int32_t>(TUPLE_REDUCE_MAX_SHAPE_LEN, 0)};
  std::vector<int64_t> reordered_fused_shape{std::vector<int64_t>(TUPLE_REDUCE_MAX_SHAPE_LEN, 0)};
  std::vector<int64_t> reduced_shape{std::vector<int64_t>(TUPLE_REDUCE_MAX_SHAPE_LEN, 0)};
  std::vector<uint32_t> map_rtoo{std::vector<uint32_t>(TUPLE_REDUCE_MAX_SHAPE_LEN, 0)};
  std::vector<int64_t> suffix_product{std::vector<int64_t>(TUPLE_REDUCE_MAX_SHAPE_LEN, 0)};
  std::vector<int64_t> reduced_prefix_product{std::vector<int64_t>(TUPLE_REDUCE_MAX_SHAPE_LEN, 0)};
  std::vector<int64_t> reduced_suffix_product{std::vector<int64_t>(TUPLE_REDUCE_MAX_SHAPE_LEN, 0)};

  private:
  bool CheckInputSize() const;
  bool GetMaxShapeLen(const ge::OpDescPtr &op_desc);
  bool GetInput();
  bool FuseAxis();
  bool EliminateTailOnes();
  bool PickScheduleStrategy();
  bool Reorder();
  bool TimeTiling_block();
  bool TimeTiling_ub();
  bool TimeTiling();
  bool SpatialTiling();
  bool SpatialTiling_block();
  bool SpatialTiling_ub();
  bool Tiling();
  bool DoTupleReduceTiling();
  bool CalcTilingKey();
  bool WriteTilingData();

}; // TupleReduce

} // TupleReduce

class TupleReduceTilingHandler : public AutoTilingHandler {
  public:
  TupleReduceTilingHandler(const std::string &op_info, const std::string &pattern, const nlohmann::json &json_info)
      : AutoTilingHandler(op_info, pattern), compileInfo(op_info, json_info) {}
  bool DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info) const override;
  bool DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info, const OpInfo &op_info) const override;
  bool ParsedSuccess() const { return compileInfo.parsed_success; };
  ~TupleReduceTilingHandler() override = default;

  private:
  const TupleReduce::TupleReduceCompileInfo compileInfo;
};
} // namespace optiling
#endif // TUPLE_REDUCE_H
