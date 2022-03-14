/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file tuple_reduce.cc
 * \brief tiling function of op
 */
#include "tuple_reduce.h"
#include <algorithm>
#include "error_log.h"
#include "tiling_handler.h"

namespace optiling {
namespace TupleReduce {
// TupleReduceCompileInfo
bool TupleReduceCompileInfo::GetCommonInfo(const std::string &op_type, const nlohmann::json &json_info) {
  common_info = json_info.at("_common_info").get<std::vector<int32_t>>();
  std::size_t expect_common_info_len = 5;
  if (common_info.size() != expect_common_info_len) { return false; }
  core_num = common_info[core_num_idx];
  ub_size = common_info[ub_size_idx];
  block_size = common_info[block_size_idx];
  atomic_support = common_info[atomic_support_idx];
  dim_var = common_info[dim_var_idx];
  return true;
}

bool TupleReduceCompileInfo::GetAxisInfo(const std::string &op_type, const nlohmann::json &json_info) {
  reduce_axis = json_info.at("_reduce_axis").get<std::vector<int32_t>>();
  disable_fuse_axes = json_info.at("_disable_fuse_axes").get<std::vector<int32_t>>();
  fused_broadcast_axis = json_info.at("_fused_broadcast_axis").get<std::vector<int32_t>>();
  fused_reduce_axis = json_info.at("_fused_reduce_axis").get<std::vector<int32_t>>();
  fusible_code = json_info.at("_fusible_code").get<std::vector<int32_t>>();
  return true;
}

bool TupleReduceCompileInfo::GetDynamicMode(const std::string &op_type, const nlohmann::json &json_info) {
  is_const = json_info.at("_is_const").get<bool>();
  runtime = json_info.at("_runtime").get<bool>();
  return true;
}
bool TupleReduceCompileInfo::GetGraphInfo(const std::string &op_type, const nlohmann::json &json_info) {
  graph_info = json_info.at("_graph_info").get<std::vector<int32_t>>();
  std::size_t expect_graph_info_len = 5;
  if (graph_info.size() != expect_graph_info_len) { return false; }
  inputs_num = graph_info[inputs_num_idx];
  buffer_count = graph_info[buffer_count_idx];
  max_dtype_size = graph_info[max_dtype_size_idx];
  min_dtype_size = graph_info[min_dtype_size_idx];
  keep_dims = graph_info[keep_dims_idx];
  shapes_length = json_info.at("_shapes_length").get<std::vector<int32_t>>();
  max_shape_len = json_info.at("_max_shape_len").get<std::int32_t>();
  each_buffer_size = (ub_size / buffer_count) / max_dtype_size;
  return true;
}

bool TupleReduceCompileInfo::CheckParseStatus(const std::string &op_type) const {
    V_OP_TILING_CHECK((core_num > 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "core_num is %d that is illegal", core_num),
                      return false);
    V_OP_TILING_CHECK((block_size > 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "block_size is %d that is illegal", block_size),
                      return false);
    return true;
}

TupleReduceCompileInfo::TupleReduceCompileInfo(const std::string &op_type, const nlohmann::json &json_info) {
  OP_LOGD(op_type.c_str(), "TupleReduceCompileInfo Constructor running");
  bool ret = GetCommonInfo(op_type, json_info);
  ret = ret && GetAxisInfo(op_type, json_info);
  ret = ret && GetDynamicMode(op_type, json_info);
  ret = ret && GetGraphInfo(op_type, json_info);
  ret = ret && CheckParseStatus(op_type);
  parsed_success = ret;
} // TupleReduceCompileInfo


// TupleReduce
bool TupleReduce::CheckInputSize() const {
  const auto &input_num = op_paras.GetInputsSize();
  V_OP_TILING_CHECK((input_num > 0 && input_num <= TUPLE_REDUCE_MAX_INPUT_NUMS),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input num is %zu that should be in (0, %zu)",
                                                   input_num, TUPLE_REDUCE_MAX_INPUT_NUMS),
                    return false);
  V_OP_TILING_CHECK((input_num == compileInfo.inputs_num),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                   "Runtime input num %zu is inconsistent with Compile time %u",
                                                   input_num, compileInfo.inputs_num),
                    return false);
  return true;
}

bool TupleReduce::GetMaxShapeLen(const ge::OpDescPtr &op_desc) {
  std::size_t max_shape_len{0};
  for (std::size_t i = 0; i < compileInfo.inputs_num; ++i) {
      const auto &cur_input_shape = op_desc->MutableInputDesc(i)->GetShape();
      std::size_t cur_shape_len = cur_input_shape.GetDimNum();
      if (cur_shape_len > max_shape_len) { max_shape_len = cur_shape_len; }
  }
  V_OP_TILING_CHECK((max_shape_len <= TUPLE_REDUCE_MAX_SHAPE_LEN),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "more than %zu dims are not supported",
                                                   TUPLE_REDUCE_MAX_SHAPE_LEN),
                    return false);
  return true;
}

bool TupleReduce::GetInput() {
  const ge::OpDescPtr &op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  V_OP_TILING_CHECK(CheckInputSize(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CheckInputSize Failed"),
                    return false);
  V_OP_TILING_CHECK(GetMaxShapeLen(op_desc),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetMaxShapeLen Failed"),
                    return false);
  // Get inputs
  for (std::uint32_t i = 0; i < compileInfo.inputs_num; ++i) {
    std::int32_t delta = compileInfo.max_shape_len - compileInfo.shapes_length[i];
    const auto &cur_input_shape = op_desc->MutableInputDesc(i)->GetShape();
    for (int32_t j = 0; j < delta; ++j) { inputs_shape[i][j] = 1; }
    for (int32_t j = delta; j < compileInfo.max_shape_len; ++j) { inputs_shape[i][j] = cur_input_shape.GetDim(j); }
  }
  return true;
}

bool TupleReduce::FuseAxis() {
  // Find target shape for Tiling
  // Preprocess, get max shape
  for (std::uint32_t i = 0; i < compileInfo.inputs_num; ++i)
    for (std::int32_t j = 0; j < compileInfo.max_shape_len; ++j)
      fused_shape[j] = (fused_shape[j] < inputs_shape[i][j]) ? inputs_shape[i][j] : fused_shape[j];
  
  if (compileInfo.is_const && !compileInfo.runtime) { // if const compile time, just resize
    std::size_t i = 0;
    while (fused_shape[i] > 0) ++i;
    fused_shape.resize(i);
    fused_shape.back() = TUPLE_REDUCE_ALIGN(fused_shape.back(), compileInfo.block_size / compileInfo.min_dtype_size);
    return true;
  }

  // else dynamic
  fused_shape.resize(compileInfo.max_shape_len);
  fused_shape.back() = TUPLE_REDUCE_ALIGN(fused_shape.back(), compileInfo.block_size / compileInfo.min_dtype_size);
  // Fuse by fusible code
  std::int32_t r_ptr = 1; // read pointer
  std::int32_t w_ptr = 0; // write pointer
  while (r_ptr < compileInfo.max_shape_len) {
    if (compileInfo.fusible_code[r_ptr] == compileInfo.fusible_code[r_ptr - 1]) {
      fused_shape[w_ptr] *= fused_shape[r_ptr++];
    } else { fused_shape[++w_ptr] = fused_shape[r_ptr++]; }
  }
  fused_shape.resize(w_ptr + 1);
  return true;
}

bool TupleReduce::EliminateTailOnes() {
  if (compileInfo.is_const && !compileInfo.runtime) { return true; }
  std::int32_t largest_element = -1;
  for (const auto &item: compileInfo.fused_reduce_axis)
    largest_element = (largest_element < item) ? item : largest_element;
  last_axis_reduce = largest_element == (std::int32_t) fused_shape.size() - 1;
  return true;
}

bool TupleReduce::PickScheduleStrategy() {
  std::size_t shape_size = fused_shape.size();
  std::int64_t reduced_tensor_numel = 1;
  std::size_t w_ptr = 0;
  reduce_one_hot.resize(shape_size);
  for (const auto &i: compileInfo.fused_reduce_axis) { reduce_one_hot[i] = 1; }
  for (std::size_t i = 0; i < shape_size; ++i) {
    if (!reduce_one_hot[i]) {
        reduced_shape[w_ptr] = fused_shape[i];
        reduced_tensor_numel *= reduced_shape[w_ptr++];
    }
  }
  reduced_shape.resize(w_ptr);
  // hard constraint: tensor numel after reduce must smaller than each buffer size
  tupleReduceTilingInfo.atomic = reduced_tensor_numel < compileInfo.each_buffer_size;
  return true;
}

bool TupleReduce::Reorder() {
  std::size_t w_ptr = 0;
  std::size_t shape_size = fused_shape.size();
  map_rtoo.resize(shape_size);
  reordered_fused_shape.resize(shape_size);

  // [R A R A R A ... *] --> [A A ... A R R ... R *]
  // map_rtoo: map reordered to original
  reordered_fused_shape.back() = fused_shape.back();
  map_rtoo.back() = --shape_size;
  // if SpatialTiling, [R A R A ...] --> [A A ... A R R ... R]
  if (!tupleReduceTilingInfo.atomic) shape_size = fused_shape.size();
  for (std::size_t i = 0; i < shape_size; ++i) {
    if (!reduce_one_hot[i]) {
        reordered_fused_shape[w_ptr] = fused_shape[i];
        map_rtoo[w_ptr++] = i;
    }
  }
  for (std::size_t i = 0; i < shape_size; ++i) {
    if (reduce_one_hot[i]) {
      reordered_fused_shape[w_ptr] = fused_shape[i];
      map_rtoo[w_ptr++] = i;
    }
  }

  return true;
}

bool TupleReduce::TimeTiling_block() {
  block_axis_lb = 0;
  block_axis_ub = reordered_fused_shape.size() - 1;
  ub_axis_lb = 0;
  ub_axis_ub = reordered_fused_shape.size() - 1;

  // block tiling
  // locate [...R R R ...] interval
  while (block_axis_lb < block_axis_ub && !reduce_one_hot[map_rtoo[block_axis_lb]]) { ++block_axis_lb; }
  while (block_axis_lb < block_axis_ub && !reduce_one_hot[map_rtoo[block_axis_ub]]) { --block_axis_ub; }
  // soft constraint: at least 1K amount of data to process per core
  tmp_product = 1;
  for (const auto &item: reduced_shape) { tmp_product *= item; }
  tmp_product *= reordered_fused_shape[block_axis_ub];
  while (block_axis_lb < block_axis_ub && tmp_product <= SINGLE_CORE_THRESHOLD) {
    tmp_product *= reordered_fused_shape[--block_axis_ub];
  }
  // prefer more core num
  tmp_product = reordered_fused_shape[block_axis_lb];
  while (block_axis_lb < block_axis_ub && tmp_product < compileInfo.core_num)
    tmp_product *= reordered_fused_shape[++block_axis_lb];
  tmp_product /= reordered_fused_shape[block_axis_lb];
  tupleReduceTilingInfo.block_tiling_axis = map_rtoo[block_axis_lb];
  tupleReduceTilingInfo.block_tiling_factor =
      TUPLE_REDUCE_CEILING(reordered_fused_shape[block_axis_lb], compileInfo.core_num / tmp_product);
  // calc block dim
  tupleReduceTilingInfo.block_dim = (std::int32_t) tmp_product
      * TUPLE_REDUCE_CEILING(reordered_fused_shape[block_axis_lb], tupleReduceTilingInfo.block_tiling_factor);

  return true;
}

bool TupleReduce::TimeTiling_ub() {
  // ub tiling
  // update reordered_fused_shape: [A A ... A R R ... R *] --> [A A ... A 1 ... 1 Ri R ... R *]
  for (std::size_t i = 0; i < block_axis_lb; ++i) if (reduce_one_hot[map_rtoo[i]]) reordered_fused_shape[i] = 1;
  reordered_fused_shape[block_axis_lb] = tupleReduceTilingInfo.block_tiling_factor;
  // hard constraint: copy gm to ub at least one number, trivial
  // hard constraint: buffer size must hold the amount of data from gm to ub
  tmp_product = 1;
  while (ub_axis_lb <= ub_axis_ub && !reduce_one_hot[map_rtoo[ub_axis_lb]]) ++ub_axis_lb;
  while (ub_axis_lb <= ub_axis_ub) tmp_product *= reordered_fused_shape[ub_axis_lb++];
  ub_axis_lb = tmp_product > compileInfo.each_buffer_size ? block_axis_lb : 0;
  tmp_product = 1;
  for (std::int32_t i = reordered_fused_shape.size() - 1; i >= 0; --i) {
    tmp_product *= reordered_fused_shape[i];
    suffix_product[i] = tmp_product;
  }
  suffix_product.resize(reordered_fused_shape.size());
  while (ub_axis_lb < ub_axis_ub && suffix_product[ub_axis_lb + 1] > compileInfo.each_buffer_size) ++ub_axis_lb;
  tupleReduceTilingInfo.ub_tiling_axis = map_rtoo[ub_axis_lb];
  // find best ub factor
  tupleReduceTilingInfo.ub_tiling_factor = compileInfo.each_buffer_size /
                                           ((ub_axis_lb < ub_axis_ub) ? suffix_product[ub_axis_lb + 1] : 1);
  tupleReduceTilingInfo.ub_tiling_factor = TUPLE_REDUCE_REFINE(fused_shape[ub_axis_lb],
                                                               tupleReduceTilingInfo.ub_tiling_factor);

  return true;
}

bool TupleReduce::TimeTiling() {
  V_OP_TILING_CHECK(TimeTiling_block(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TimeTiling_block Failed"), return false);
  V_OP_TILING_CHECK(TimeTiling_ub(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TimeTiling_ub Failed"), return false);
  return true;
}

bool TupleReduce::SpatialTiling_block() {
  block_axis_lb = 0;
  block_axis_ub = reduced_shape.size() - 1;
  core_num = 1;
  ub_axis_lb = 0;
  ub_axis_ub = reordered_fused_shape.size() - 1;
  // reordered: [A A ... A R R ... R]

  // block tiling
  // hard constraint: tensor numel after reduce on every block must greater than block size
  tmp_product = reduced_shape[block_axis_ub];
  while (block_axis_lb < block_axis_ub && tmp_product < MINIMUM_OVERLAP_COEFFICIENT)
    tmp_product *= reduced_shape[--block_axis_ub];
  // hard constraint: tensor numel after reduce on every block should smaller than buffer size
  tmp_product = 1;
  for (const auto &item: reduced_shape) tmp_product *= item;
  tmp_product /= reduced_shape[block_axis_lb];
  while (block_axis_lb < block_axis_ub && tmp_product > compileInfo.each_buffer_size)
    tmp_product /= reduced_shape[++block_axis_lb];
  tupleReduceTilingInfo.block_tiling_axis = map_rtoo[block_axis_lb];
  // find best block factor
  tmp_product = 1;
  for (std::size_t i = 0; i < block_axis_lb; ++i) tmp_product *= reduced_shape[i];
  tupleReduceTilingInfo.block_tiling_factor
    = TUPLE_REDUCE_CEILING(reduced_shape[block_axis_lb], TUPLE_REDUCE_CEILING(compileInfo.core_num, tmp_product));
  // hard constraint: tensor numel after reduce on every block must greater than block size
  tmp_product = tupleReduceTilingInfo.block_tiling_factor;
  for (std::size_t i = reduced_shape.size() - 1; i > block_axis_lb; --i) tmp_product *= reduced_shape[i];
  if (tmp_product < MINIMUM_OVERLAP_COEFFICIENT) {
    tmp_product /= tupleReduceTilingInfo.block_tiling_factor;
    tmp_product *= reduced_shape[block_axis_lb];
    core_num = TUPLE_REDUCE_CEILING(tmp_product, SINGLE_CORE_THRESHOLD);
    tmp_product /= reduced_shape[block_axis_lb];
    tupleReduceTilingInfo.block_tiling_factor = TUPLE_REDUCE_CEILING(reduced_shape[block_axis_lb],
                                                                     TUPLE_REDUCE_CEILING(core_num, tmp_product));
  }
  // calc block dim
  tmp_product = 1;
  for (std::size_t i = 0; i < block_axis_lb; ++i) tmp_product *= reduced_shape[i];
  tupleReduceTilingInfo.block_dim = (std::int32_t) tmp_product
      * (std::int32_t) TUPLE_REDUCE_CEILING(reduced_shape[block_axis_lb], tupleReduceTilingInfo.block_tiling_factor);
  
  return true;
}

bool TupleReduce::SpatialTiling_ub() {
  // ub tiling
  // find block_axis_lb in reordered_fused_shape
  ub_axis_lb = block_axis_lb;
  // hard constraint: copy gm to ub at least one number, trivial
  // hard constraint: buffer size must hold the amount of data from gm to ub
  tmp_product = 1;
  for (std::size_t i = reordered_fused_shape.size() - 1; i > block_axis_lb; --i)
    tmp_product *= reordered_fused_shape[i];
  while (ub_axis_lb < ub_axis_ub
      && tmp_product * (compileInfo.block_size / compileInfo.min_dtype_size) > compileInfo.each_buffer_size)
    tmp_product /= reordered_fused_shape[++ub_axis_lb];
  tupleReduceTilingInfo.ub_tiling_axis = map_rtoo[ub_axis_lb];
  // find best ub factor
  tupleReduceTilingInfo.ub_tiling_factor =
      ((compileInfo.each_buffer_size / tmp_product) / (compileInfo.block_size / compileInfo.min_dtype_size))
          * (compileInfo.block_size / compileInfo.min_dtype_size);
  if (tupleReduceTilingInfo.ub_tiling_factor > reordered_fused_shape[ub_axis_lb]) {
    tupleReduceTilingInfo.ub_tiling_factor = reordered_fused_shape[ub_axis_lb];
  }
  if (tupleReduceTilingInfo.ub_tiling_axis == tupleReduceTilingInfo.block_tiling_axis) {
    if (tupleReduceTilingInfo.ub_tiling_factor > tupleReduceTilingInfo.block_tiling_factor) {
      tupleReduceTilingInfo.ub_tiling_factor = tupleReduceTilingInfo.block_tiling_factor;
    }
    tupleReduceTilingInfo.ub_tiling_factor = (compileInfo.block_size / compileInfo.min_dtype_size)
        * TUPLE_REDUCE_REFINE(tupleReduceTilingInfo.block_tiling_factor 
                              / (compileInfo.block_size / compileInfo.min_dtype_size),
                              tupleReduceTilingInfo.ub_tiling_factor
                              / (compileInfo.block_size / compileInfo.min_dtype_size));
  } else {
    tupleReduceTilingInfo.ub_tiling_factor = (compileInfo.block_size / compileInfo.min_dtype_size)
        * TUPLE_REDUCE_REFINE(reordered_fused_shape[ub_axis_lb]
                              / (compileInfo.block_size / compileInfo.min_dtype_size),
                              tupleReduceTilingInfo.ub_tiling_factor
                              / (compileInfo.block_size / compileInfo.min_dtype_size));
  }

  return true;
}

bool TupleReduce::SpatialTiling() {
  V_OP_TILING_CHECK(SpatialTiling_block(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SpatialTiling_block Failed"), return false);
  V_OP_TILING_CHECK(SpatialTiling_ub(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SpatialTiling_ub Failed"), return false);
  return true;
}

bool TupleReduce::Tiling() {
  V_OP_TILING_CHECK(Reorder(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Reorder Failed"), return false);
  if (tupleReduceTilingInfo.atomic) {
    V_OP_TILING_CHECK(TimeTiling(),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TimeTiling Failed"), return false);
  } else {
    V_OP_TILING_CHECK(SpatialTiling(),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SpatialTiling Failed"), return false);
  }
  return true;
}

bool TupleReduce::DoTupleReduceTiling() {
  /* Situations of DoTiling include:
     1. dynamic
     2. const compile time
     3. const runtime
  */

  V_OP_TILING_CHECK(GetInput(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetInput Failed"), return false);
  V_OP_TILING_CHECK(FuseAxis(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "FuseAxis Failed"), return false);
  V_OP_TILING_CHECK(EliminateTailOnes(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "EliminateTailOnes Failed"),
                    return false);
  V_OP_TILING_CHECK(PickScheduleStrategy(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "PickScheduleStrategy Failed"),
                    return false);
  V_OP_TILING_CHECK(Tiling(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Tiling Failed"), return false);
  
  return true;
}

bool TupleReduce::CalcTilingKey() {
  std::int64_t base = 256;
  std::int64_t reduce_pattern = 0;
  std::int64_t reduce_pattern_keybase = 1000;
  std::int64_t atomic_keybase = 100;
  std::int64_t block_tiling_axis_keybase = 10;
  std::int64_t ub_tiling_axis_keybase = 1;
  for (const auto &item: reduce_one_hot) reduce_pattern += (std::int64_t) item * (base >>= 1);
  // [...][.][.][.] == [reduce_pattern][schedule_type][block][ub]
  tupleReduceTilingInfo.tiling_key
      = reduce_pattern * reduce_pattern_keybase
      + tupleReduceTilingInfo.atomic * atomic_keybase
      + tupleReduceTilingInfo.block_tiling_axis * block_tiling_axis_keybase
      + tupleReduceTilingInfo.ub_tiling_axis * ub_tiling_axis_keybase;

  return true;
}

bool TupleReduce::WriteTilingData() {
  V_OP_TILING_CHECK(CalcTilingKey(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CalcTilingKey Failed"), return false);
  run_info.SetClearAtomic(tupleReduceTilingInfo.atomic);
  run_info.SetBlockDim(static_cast<uint32_t>(tupleReduceTilingInfo.block_dim));
  run_info.SetTilingKey(static_cast<uint32_t>(tupleReduceTilingInfo.tiling_key));
  // if not runtime, return true
  if (compileInfo.is_const && compileInfo.runtime) { return true; }
  // add _dim_*
  std::int32_t dim_var = compileInfo.dim_var;
  for (const auto &item: fused_shape) {
    if (dim_var & 1) run_info.AddTilingData((int32_t) item);
    dim_var >>= 1;
  }
  // add _block_factor && _ub_factor
  run_info.AddTilingData((int32_t) tupleReduceTilingInfo.block_tiling_factor);
  run_info.AddTilingData((int32_t) tupleReduceTilingInfo.ub_tiling_factor);

  return true;
}

bool TupleReduce::DoTiling() {
  bool ret = DoTupleReduceTiling();
  ret = ret && WriteTilingData();
  return ret;
}
} // namespace TupleReduce


// TupleReduceTilingHandler
bool TupleReduceTilingHandler::DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info) const {
  OP_LOGD(op_type.c_str(), "tiling running");
  TupleReduce::TupleReduce tuple_reduce(op_type, op_paras, compileInfo, run_info);
  return tuple_reduce.DoTiling();
}

bool TupleReduceTilingHandler::DoTiling(const ge::Operator &op_paras, utils::OpRunInfo &run_info,
                                        const OpInfo &op_info) const {
  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Tuple reduce custom tiling is not supported yet");
  return false;
}

std::shared_ptr<AutoTilingHandler> CreateTupleReduceTilingHandler(const std::string &op_type,
                                                                  const std::string &pattern,
                                                                  const nlohmann::json &parsed_compile_info) {
  auto CompileInfo = std::make_shared<TupleReduceTilingHandler>(op_type, pattern, parsed_compile_info);
  return CompileInfo->ParsedSuccess() ? CompileInfo : std::shared_ptr<AutoTilingHandler>(nullptr);
} // TupleReduceTilingHandler
} // namespace optiling
