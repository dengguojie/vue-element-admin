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
bool TupleReduceCompileInfo::GetCommonInfo(const std::string& op_type, const nlohmann::json& json_info) {
  int32_t idx = 0;
  if (json_info.contains("_common_info")) {
    common_info = json_info.at("_common_info").get<std::vector<int32_t>>();
    core_num = common_info[idx];
    block_size = common_info[++idx];
    atomic_support = common_info[++idx];
    atomic_threshold = common_info[++idx];
  }

  return true;
}

bool TupleReduceCompileInfo::GetAxisInfo(const std::string& op_type, const nlohmann::json& json_info) {
  if (json_info.contains("_fused_reduce_axis")) {
    fused_reduce_axis = json_info.at("_fused_reduce_axis").get<std::vector<int32_t>>();
  }
  if (json_info.contains("_fusible_code")) {
    fusible_code = json_info.at("_fusible_code").get<std::vector<int32_t>>();
  }
  if (json_info.contains("_reduce_axis")) {
    reduce_axis = json_info.at("_reduce_axis").get<std::vector<int32_t>>();
  }

  return true;
}

bool TupleReduceCompileInfo::GetDynamicMode(const std::string& op_type, const nlohmann::json& json_info) {
  if (json_info.contains("_is_const")) {
    is_const = json_info.at("_is_const").get<bool>();
  }
  if (json_info.contains("_runtime")) {
    runtime = json_info.at("_runtime").get<bool>();
  }
  if (json_info.contains("_dim_var_code")) {
    const auto& local_dim_var_code = json_info.at("_dim_var_code").get<std::unordered_map<std::string, int32_t>>();
    for (const auto& item: local_dim_var_code) {
      dim_var_code[std::stoi(item.first)] = item.second;
    }
  }

  return true;
}

bool TupleReduceCompileInfo::GetGraphInfo(const std::string& op_type, const nlohmann::json& json_info) {
  int32_t idx = 0;
  if (json_info.contains("_graph_info")) {
    graph_info = json_info.at("_graph_info").get<std::vector<int32_t>>();
    inputs_num = graph_info[idx];
    min_dtype_size = graph_info[++idx];
    max_dtype_size = graph_info[++idx];
    reduce_dtype_size = graph_info[++idx];
  }
  if (json_info.contains("_buffer_size")) {
    buffer_size = json_info.at("_buffer_size").get<std::vector<int32_t>>();
  }

  return true;
}

TupleReduceCompileInfo::TupleReduceCompileInfo(const std::string& op_type, const nlohmann::json& json_info) {
  OP_LOGD(op_type.c_str(), "TupleReduceCompileInfo Constructor running");
  bool ret = GetCommonInfo(op_type, json_info);
  ret = ret && GetAxisInfo(op_type, json_info);
  ret = ret && GetDynamicMode(op_type, json_info);
  ret = ret && GetGraphInfo(op_type, json_info);
  parsed_success = ret;
}  // TupleReduceCompileInfo

// TupleReduce
bool TupleReduce::GetInput() {
  // Get op_desc
  const ge::OpDescPtr& op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);

  // Get max shape length
  for (std::size_t i = 0; i < compileInfo.inputs_num; ++i) {
    const auto& cur_input_shape = op_desc->MutableInputDesc(i)->GetShape();
    size_t cur_shape_len = cur_input_shape.GetDimNum();
    max_shape_len = cur_shape_len > max_shape_len ? cur_shape_len : max_shape_len;
  }

  // Get inputs
  for (std::uint32_t i = 0; i < compileInfo.inputs_num; ++i) {
    const auto& cur_input_shape = op_desc->MutableInputDesc(i)->GetShape();
    size_t cur_shape_len = cur_input_shape.GetDimNum();
    int32_t delta = static_cast<int32_t>(max_shape_len - cur_shape_len);
    for (int32_t j = 0; j < static_cast<int32_t>(max_shape_len); ++j) {
      inputs_shape[i][j] = j < delta ? 1 : cur_input_shape.GetDim(j);
    }
  }

  return true;
}

bool TupleReduce::FuseAxis() {
  // Find target shape for Tiling
  // Preprocess, get max shape
  for (std::uint32_t i = 0; i < compileInfo.inputs_num; ++i) {
    for (std::int32_t j = 0; j < static_cast<int32_t>(max_shape_len); ++j) {
      fused_shape[j] = (fused_shape[j] < inputs_shape[i][j]) ? inputs_shape[i][j] : fused_shape[j];
    }
  }

  if (compileInfo.is_const && !compileInfo.runtime) {  // if const compile time, just resize
    size_t i = 0;
    while (fused_shape[i] > 0) ++i;
    fused_shape.resize(i);
    return true;
  }

  // Fuse by fusible code
  std::int32_t r_ptr = 1;  // read pointer
  std::int32_t w_ptr = 0;  // write pointer
  while (r_ptr < static_cast<int32_t>(max_shape_len)) {
    if (compileInfo.fusible_code[r_ptr] == compileInfo.fusible_code[r_ptr - 1]) {
      fused_shape[w_ptr] *= fused_shape[r_ptr++];
    } else {
      fused_shape[++w_ptr] = fused_shape[r_ptr++];
    }
  }
  fused_shape.resize(w_ptr + 1);

  return true;
}

bool TupleReduce::EliminateTailOnes() {
  int32_t largest_element = fused_shape.back();
  for (const auto& item : compileInfo.fused_reduce_axis) {
    last_axis_reduce = last_axis_reduce || fused_shape.size() -1 == static_cast<size_t>(item);
  }
  if (compileInfo.runtime && largest_element == 1 && !last_axis_reduce) {
    fused_shape.resize(fused_shape.size() - 1);
  }
  last_axis_align = fused_shape.back() == TUPLE_REDUCE_ALIGN(fused_shape.back(),
                                                             compileInfo.block_size / compileInfo.min_dtype_size);
  fused_shape.back() = TUPLE_REDUCE_ALIGN(fused_shape.back(),
                                          compileInfo.block_size / compileInfo.min_dtype_size);

  return true;
}

bool TupleReduce::PickScheduleStrategy() {
  std::size_t shape_size = fused_shape.size();
  std::int64_t reduced_tensor_numel = 1;
  std::size_t w_ptr = 0;
  std::int32_t a_threshold = 60;
  reduce_one_hot.resize(shape_size);
  for (const auto& i : compileInfo.fused_reduce_axis) {
    reduce_one_hot[i] = 1;
  }
  for (std::size_t i = 0; i < shape_size; ++i) {
    if (!reduce_one_hot[i]) {
      reduced_shape[w_ptr] = fused_shape[i];
      reduced_tensor_numel *= reduced_shape[w_ptr++];
    }
  }
  reduced_shape.resize(w_ptr);
  // hard constraint: atomic is support
  tupleReduceTilingInfo.atomic = compileInfo.atomic_support;
  // hard constraint: tensor numel after reduce must smaller than each buffer size
  // soft constraint: tensor numel after reduce should smaller than core_num * block_size / reduce_dtype_size
  tupleReduceTilingInfo.atomic = tupleReduceTilingInfo.atomic
                                 && (reduced_tensor_numel < compileInfo.buffer_size[0])
                                 && (reduced_tensor_numel <= compileInfo.atomic_threshold)
                                 && (reduced_shape[0] < a_threshold);

  return true;
}

bool TupleReduce::ReducePattern() {
  int64_t bin = 2;
  int64_t dec = 10;
  reduce_pattern = 0;
  for(auto item: reduce_one_hot) {
    reduce_pattern = bin * reduce_pattern + static_cast<int32_t>(item);
  }
  reduce_pattern = dec * reduce_pattern + static_cast<int64_t>(reduce_one_hot.size());

  return true;
}

bool TupleReduce::PickOptMode() {
  int64_t transpose_reduce_threshold = 128;
  int64_t AR_Pattern = 12;
  int64_t RA_Pattern = 22;
  int64_t normal_mode = 0;
  int64_t transpose_reduce_mode = 1;
  int64_t align_pad_mode = 2;

  V_OP_TILING_CHECK(ReducePattern(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ReducePattern Failed"),
                    return false);
  // if not last axis align, enable opt
  // if pattern is AR and R < 128, transpose reduce, buffer_size[1]
  // else align pad
  if (!last_axis_align && static_cast<int64_t>(fused_shape.back()) < transpose_reduce_threshold) {
    if (reduce_pattern == AR_Pattern) { // transpose reduce
      opt_mode = transpose_reduce_mode;
    } else if (reduce_pattern == RA_Pattern) { // align pad
      opt_mode = align_pad_mode;
    }
  } else {
    opt_mode = normal_mode;
  }

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
  while (block_axis_lb < block_axis_ub && !reduce_one_hot[map_rtoo[block_axis_lb]]) {
    ++block_axis_lb;
  }
  while (block_axis_lb < block_axis_ub && !reduce_one_hot[map_rtoo[block_axis_ub]]) {
    --block_axis_ub;
  }
  // soft constraint: at least 1K amount of data to process per core
  tmp_product = 1;
  for (const auto& item : reduced_shape) {
    tmp_product *= item;
  }
  tmp_product *= reordered_fused_shape[block_axis_ub];
  while (block_axis_lb < block_axis_ub && tmp_product <= SINGLE_CORE_THRESHOLD) {
    tmp_product *= reordered_fused_shape[--block_axis_ub];
  }
  // prefer more core num
  tmp_product = reordered_fused_shape[block_axis_lb];
  while (block_axis_lb < block_axis_ub && tmp_product < compileInfo.core_num) {
    tmp_product *= reordered_fused_shape[++block_axis_lb];
  }
  tmp_product /= reordered_fused_shape[block_axis_lb];
  tupleReduceTilingInfo.block_tiling_axis = map_rtoo[block_axis_lb];
  tupleReduceTilingInfo.block_tiling_factor = TUPLE_REDUCE_CEILING(reordered_fused_shape[block_axis_lb],
                                                                   compileInfo.core_num / tmp_product);
  // calc block dim
  tupleReduceTilingInfo.block_dim = static_cast<int32_t>(tmp_product)
                                    * TUPLE_REDUCE_CEILING(reordered_fused_shape[block_axis_lb],
                                                           tupleReduceTilingInfo.block_tiling_factor);

  return true;
}

bool TupleReduce::TimeTiling_ub() {
  // ub tiling
  // update reordered_fused_shape: [A A ... A R R ... R *] --> [A A ... A 1 ... 1 Ri R ... R *]
  for (std::size_t i = 0; i < block_axis_lb; ++i) {
    if (reduce_one_hot[map_rtoo[i]]) {
      reordered_fused_shape[i] = 1;
    }
  }
  reordered_fused_shape[block_axis_lb] = tupleReduceTilingInfo.block_tiling_factor;
  // hard constraint: copy gm to ub at least one number, trivial
  // hard constraint: buffer size must hold the amount of data from gm to ub
  tmp_product = 1;
  while (ub_axis_lb <= ub_axis_ub && !reduce_one_hot[map_rtoo[ub_axis_lb]]) {
    ++ub_axis_lb;
  }
  while (ub_axis_lb <= ub_axis_ub) {
    tmp_product *= reordered_fused_shape[ub_axis_lb++];
  }
  ub_axis_lb = tmp_product > compileInfo.buffer_size[opt_mode] ? block_axis_lb : 0;

  tmp_product = 1;
  for (std::int32_t i = reordered_fused_shape.size() - 1; i >= 0; --i) {
    tmp_product *= reordered_fused_shape[i];
    suffix_product[i] = tmp_product;
  }
  suffix_product.resize(reordered_fused_shape.size());
  while (ub_axis_lb < ub_axis_ub && suffix_product[ub_axis_lb + 1] > compileInfo.buffer_size[opt_mode]) {
    ++ub_axis_lb;
  }
  tupleReduceTilingInfo.ub_tiling_axis = map_rtoo[ub_axis_lb];

  // find best ub factor
  tupleReduceTilingInfo.ub_tiling_factor =
    compileInfo.buffer_size[opt_mode] / ((ub_axis_lb < ub_axis_ub) ? suffix_product[ub_axis_lb + 1] : 1);
  tupleReduceTilingInfo.ub_tiling_factor = TUPLE_REDUCE_REFINE(reordered_fused_shape[ub_axis_lb],
                                                               tupleReduceTilingInfo.ub_tiling_factor);

  return true;
}

bool TupleReduce::TimeTiling() {
  V_OP_TILING_CHECK(TimeTiling_block(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TimeTiling_block Failed"),
                    return false);
  V_OP_TILING_CHECK(TimeTiling_ub(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TimeTiling_ub Failed"),
                    return false);

  return true;
}

bool TupleReduce::SpatialTiling_block() {
  block_axis_lb = 0;
  block_axis_ub = reduced_shape.size() - 1;

  // block tiling
  // hard constraint: tensor numel after reduce on every block must greater than block size
  tmp_product = 1;
  while (block_axis_lb < block_axis_ub && tmp_product * compileInfo.reduce_dtype_size < compileInfo.block_size) {
    tmp_product *= reduced_shape[block_axis_ub--];
  }
  // hard constraint: tensor numel after reduce on every block should smaller than buffer size
  tmp_product = 1;
  for (const auto& item : reduced_shape) {
    tmp_product *= item;
  }
  while (block_axis_lb < block_axis_ub
         && tmp_product > compileInfo.buffer_size[opt_mode] * reduced_shape[block_axis_lb]) {
    tmp_product /= reduced_shape[block_axis_lb++];
  }
  if (!last_axis_reduce && block_axis_lb == reduced_shape.size() - 1) {
    tupleReduceTilingInfo.block_tiling_axis = map_rtoo[reordered_fused_shape.size() - 1];
  } else {
    tupleReduceTilingInfo.block_tiling_axis = map_rtoo[block_axis_lb];
  }

  // find best block factor
  tmp_product = 1;
  for (std::size_t i = 0; i < block_axis_lb; ++i) {
    tmp_product *= reduced_shape[i];
  }
  tupleReduceTilingInfo.block_tiling_factor = TUPLE_REDUCE_CEILING(reduced_shape[block_axis_lb],
                                                                   TUPLE_REDUCE_CEILING(compileInfo.core_num,
                                                                                        tmp_product));

  // calc block dim
  tupleReduceTilingInfo.block_dim = (std::int32_t)tmp_product
                                  * (std::int32_t)TUPLE_REDUCE_CEILING(reduced_shape[block_axis_lb],
                                                                       tupleReduceTilingInfo.block_tiling_factor);

  return true;
}

bool TupleReduce::SpatialTiling_ub() {
  ub_axis_lb = block_axis_lb;
  ub_axis_ub = reordered_fused_shape.size() - 1;
  // reordered: [A A ... A R R ... R *]

  // ub tiling
  // hard constraint: copy gm to ub at least one number, trivial
  // hard constraint: buffer size must hold the amount of data from gm to ub
  tmp_product = 1;
  for (std::size_t i = reordered_fused_shape.size() - 1; i > block_axis_lb; --i) {
    tmp_product *= reordered_fused_shape[i];
  }
  while (ub_axis_lb < ub_axis_ub && tmp_product > compileInfo.buffer_size[opt_mode]) {
    tmp_product /= reordered_fused_shape[++ub_axis_lb];
  }
  tupleReduceTilingInfo.ub_tiling_axis = map_rtoo[ub_axis_lb];

  // find best ub factor
  tupleReduceTilingInfo.ub_tiling_factor = compileInfo.buffer_size[opt_mode] / tmp_product;
  if (tupleReduceTilingInfo.ub_tiling_axis == tupleReduceTilingInfo.block_tiling_axis) {
    if (tupleReduceTilingInfo.ub_tiling_factor > tupleReduceTilingInfo.block_tiling_factor) {
      tupleReduceTilingInfo.ub_tiling_factor = tupleReduceTilingInfo.block_tiling_factor;
    }
    tupleReduceTilingInfo.ub_tiling_factor = TUPLE_REDUCE_REFINE(tupleReduceTilingInfo.block_tiling_factor,
                                                                 tupleReduceTilingInfo.ub_tiling_factor);
  } else {
    tupleReduceTilingInfo.ub_tiling_factor = TUPLE_REDUCE_REFINE(reordered_fused_shape[ub_axis_lb],
                                                                 tupleReduceTilingInfo.ub_tiling_factor);
  }

  return true;
}

bool TupleReduce::SpatialTiling() {
  V_OP_TILING_CHECK(SpatialTiling_block(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SpatialTiling_block Failed"),
                    return false);
  V_OP_TILING_CHECK(SpatialTiling_ub(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SpatialTiling_ub Failed"),
                    return false);

  return true;
}

bool TupleReduce::Tiling() {
  V_OP_TILING_CHECK(Reorder(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Reorder Failed"),
                    return false);
  if (tupleReduceTilingInfo.atomic) {
    V_OP_TILING_CHECK(TimeTiling(),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TimeTiling Failed"),
                      return false);
  } else {
    V_OP_TILING_CHECK(SpatialTiling(),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SpatialTiling Failed"),
                      return false);
  }

  return true;
}

bool TupleReduce::DoTupleReduceTiling() {
  /* Situations of DoTiling include:
     1. dynamic
     2. const compile time
     3. const runtime
  */
  V_OP_TILING_CHECK(GetInput(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetInput Failed"),
                    return false);
  V_OP_TILING_CHECK(FuseAxis(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "FuseAxis Failed"),
                    return false);
  V_OP_TILING_CHECK(EliminateTailOnes(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "EliminateTailOnes Failed"),
                    return false);
  V_OP_TILING_CHECK(PickScheduleStrategy(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "PickScheduleStrategy Failed"),
                    return false);
  V_OP_TILING_CHECK(PickOptMode(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "PickOptMode Failed"),
                    return false);
  V_OP_TILING_CHECK(Tiling(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Tiling Failed"),
                    return false);

  return true;
}

bool TupleReduce::CalcTilingKey() {
  int64_t dec = 10;
  // [...][.][.][.] == [reduce_pattern][opt_mode][atomic][block][ub]
  tupleReduceTilingInfo.tiling_key = reduce_pattern;
  tupleReduceTilingInfo.tiling_key = tupleReduceTilingInfo.tiling_key * dec + opt_mode;
  tupleReduceTilingInfo.tiling_key = tupleReduceTilingInfo.tiling_key * dec + tupleReduceTilingInfo.atomic;
  tupleReduceTilingInfo.tiling_key = tupleReduceTilingInfo.tiling_key * dec + tupleReduceTilingInfo.block_tiling_axis;
  tupleReduceTilingInfo.tiling_key = tupleReduceTilingInfo.tiling_key * dec + tupleReduceTilingInfo.ub_tiling_axis;

  return true;
}

bool TupleReduce::WriteTilingData() {
  V_OP_TILING_CHECK(CalcTilingKey(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CalcTilingKey Failed"), return false);
  run_info.SetClearAtomic(tupleReduceTilingInfo.atomic);
  run_info.SetBlockDim(static_cast<uint32_t>(tupleReduceTilingInfo.block_dim));
  run_info.SetTilingKey(static_cast<uint32_t>(tupleReduceTilingInfo.tiling_key));
  // if not runtime, return true
  if (compileInfo.is_const && compileInfo.runtime) {
    return true;
  }
  // add _dim_*
  std::int32_t dim_var = compileInfo.dim_var_code.at(reduce_pattern);
  for (const auto& item : fused_shape) {
    if (dim_var & 1)
      run_info.AddTilingData(static_cast<int32_t>(item));
    dim_var >>= 1;
  }
  // add _block_factor && _ub_factor
  run_info.AddTilingData(static_cast<int32_t>(tupleReduceTilingInfo.block_tiling_factor));
  run_info.AddTilingData(static_cast<int32_t>(tupleReduceTilingInfo.ub_tiling_factor));

  return true;
}

bool TupleReduce::DoTiling() {
  bool ret = DoTupleReduceTiling();
  ret = ret && WriteTilingData();
  return ret;
}
}  // namespace TupleReduce

// TupleReduceTilingHandler
bool TupleReduceTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "tiling running");
  TupleReduce::TupleReduce tuple_reduce(op_type, op_paras, compileInfo, run_info);
  return tuple_reduce.DoTiling();
}

bool TupleReduceTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                        const OpInfo& op_info) const {
  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Tuple reduce custom tiling is not supported yet");
  return false;
}

std::shared_ptr<AutoTilingHandler> CreateTupleReduceTilingHandler(const std::string& op_type,
                                                                  const std::string& pattern,
                                                                  const nlohmann::json& parsed_compile_info) {
  auto CompileInfo = std::make_shared<TupleReduceTilingHandler>(op_type, pattern, parsed_compile_info);
  return CompileInfo->ParsedSuccess() ? CompileInfo : std::shared_ptr<AutoTilingHandler>(nullptr);
}  // TupleReduceTilingHandler
}  // namespace optiling
