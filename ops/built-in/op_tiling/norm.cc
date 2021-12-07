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
 * \file norm.cc
 * \brief tiling function of op
 */
#include <algorithm>
#include <numeric>
#include "norm.h"
#include "error_log.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
namespace {
  bool IsElementInVector(const std::vector<int32_t>& shape, const int32_t& value) {
    for (const auto& single_dim : shape) {
      if (single_dim == value) {
        return true;
      }
    }
    return false;
  }

  int64_t CalcAfterReduceShapeProduct(const std::vector<int64_t>& shape, const std::vector<int32_t>& axis) {
    int64_t result = 1;
    for (std::size_t i = 0; i < shape.size(); i++) {
      if (!IsElementInVector(axis, i)) {
        result = result * shape[i];
      }
    }
    return result;
  }

  int64_t CalcReduceShapeProduct(const std::vector<int64_t>& shape, const std::vector<int32_t>& axis) {
    int64_t result = 1;
    for (std::size_t i = 0; i < shape.size(); i++) {
      if (IsElementInVector(axis, i)) {
        result = result * shape[i];
      }
    }
    return result;
  }

  int32_t CalcPattern(const std::vector<int64_t>& shape, const std::vector<int32_t>& axis) {
    int32_t pattern = 0;
    uint32_t special_axis_weight = 2;
    for (std::size_t i = 0; i < shape.size(); i++) {
      if (IsElementInVector(axis, i)) {
        pattern += special_axis_weight << (shape.size() - i - 1);
      } else {
        pattern += (int32_t)(shape.size() - special_axis_weight - i) >= 0 ?
                  special_axis_weight << (shape.size() - special_axis_weight - i) :
                  1;
      }
    }
    return pattern;
  }

  int64_t CeilDiv(const int64_t& dividend, const int64_t& divisor) {
    V_OP_TILING_CHECK((divisor != 0),
                      VECTOR_INNER_ERR_REPORT_TILIING("Norm", "In CeilDiv func, divisor cannot be zero."),
                      return divisor);
    return (dividend + divisor - 1) / divisor;
  }

  int64_t CalcRemainder(const int64_t& dividend, const int64_t& divisor) {
    V_OP_TILING_CHECK((divisor != 0),
                      VECTOR_INNER_ERR_REPORT_TILIING("Norm", "In CalcRemainder func, divisor cannot be zero."),
                      return divisor);
    return dividend % divisor == 0 ? divisor : dividend % divisor;
  }
}

bool Norm::GetInput() {
  const auto& input_num = op_paras.GetInputsSize();
  V_OP_TILING_CHECK((input_num > 0 && input_num <= NORM_MAX_INPUT_NUMS),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input num is %zu that is larger than max nums %zu",
                                                    input_num, NORM_MAX_INPUT_NUMS),
                    return false);
  V_OP_TILING_CHECK((compileInfo.input_type.size() == input_num),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                    "size of input type %zu should be equal to input num %zu",
                                                    compileInfo.input_type.size(), input_num),
                    return false);

  std::array<bool, NORM_MAX_INPUT_NUMS> input_flag{false};
  std::array<int64_t, NORM_MAX_INPUT_NUMS> max_shape{1};
  const ge::OpDescPtr& op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);

  // obtain max dim len
  for (std::size_t i = 0; i < compileInfo.input_type.size(); i++) {
    const auto& cur_input_shape = op_desc->MutableInputDesc(i)->GetShape();
    std::size_t cur_dim_len = cur_input_shape.GetDimNum();
    V_OP_TILING_CHECK((cur_dim_len <= NORM_MAX_DIM_LEN),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "more than %zu dims are not supported",
                                                      NORM_MAX_DIM_LEN),
                      return false);
    if (cur_dim_len > max_dim_len) {
      max_dim_len = cur_dim_len;
    }
  }
  // get before broadcast input (input type is not 0)
  // max dim len = after broadcast shape len
  for (std::size_t i = 0; i < compileInfo.input_type.size(); i++) {
    const int32_t& single_input_type = compileInfo.input_type[i];
    // this input type has been recorded
    if (input_flag[single_input_type]) {
      continue;
    }
    const auto& cur_input_shape = op_desc->MutableInputDesc(i)->GetShape();
    std::size_t cur_dim_len = cur_input_shape.GetDimNum();
    input_flag[single_input_type] = true;
    if (single_input_type == 0) {
      // get after broadcast shape(input type is 0)
      V_OP_TILING_CHECK((cur_dim_len == max_dim_len),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                        "the shape length of after broadcast input %zu should be"
                                                        "euqal to max dim len %zu",
                                                        cur_dim_len, max_dim_len),
                        return false);
      for (std::size_t j = 0; j < max_dim_len; j++) {
        input_shape_ori[j] = cur_input_shape.GetDim(j);
      }
      input_shape_ori.resize(max_dim_len);
    } else {
      // get before broadcast shape(input type is not 0)
      V_OP_TILING_CHECK((cur_dim_len <= max_dim_len),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                        "the shape length of before broadcast input %zu should be"
                                                        "less than or euqal to max dim len %zu",
                                                        cur_dim_len, max_dim_len),
                        return false);
      std::size_t diff_len = max_dim_len - cur_dim_len;
      for (std::size_t j = 0; j < max_dim_len; j++) {
        if (j < diff_len) {
          // padding 1
          before_broadcast_shapes[before_broadcast_input_num][j] = 1;
        } else {
          int64_t single_dim = cur_input_shape.GetDim(j - diff_len);
          if (single_dim > max_shape[j]) {
            max_shape[j] = single_dim;
          }
          before_broadcast_shapes[before_broadcast_input_num][j] = single_dim;
        }
      }
      before_broadcast_input_num++;
    }
  }

  // dont have after broadcast input
  if (!input_flag[0]) {
    for (std::size_t j = 0; j < max_dim_len; j++) {
      input_shape_ori[j] = max_shape[j];
    }
    input_shape_ori.resize(max_dim_len);
  }

  return true;
}

bool Norm::Init() {
  // init reduce axis
  std::size_t ori_reduce_axis_len = compileInfo.ori_reduce_axis.size();
  for (std::size_t i = 0; i < ori_reduce_axis_len; i++) {
    int32_t single_reduce_axis = compileInfo.ori_reduce_axis[i];
    // convert reduce axis (-1 -> max_dim_len - 1)
    if (single_reduce_axis < 0) {
      single_reduce_axis = max_dim_len + single_reduce_axis;
    }
    // check reduce axis value
    V_OP_TILING_CHECK((single_reduce_axis < (int32_t)max_dim_len),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of reduce axis %d is illegal",
                                                      single_reduce_axis),
                      return false);
    reduce_axis_ori[i] = single_reduce_axis;
  }
  reduce_axis_ori.resize(ori_reduce_axis_len);

  // init broadcast axis
  if (compileInfo.is_broadcast_axis_known) {
    std::size_t ori_broadcast_axis_len = compileInfo.ori_broadcast_axis.size();
    for (std::size_t i = 0; i < ori_broadcast_axis_len; i++) {
      int32_t single_broadcast_axis = compileInfo.ori_broadcast_axis[i];
      // convert broadcast axis (-1 -> max_dim_len - 1)
      if (single_broadcast_axis < 0) {
        single_broadcast_axis = max_dim_len + single_broadcast_axis;
      }
      // check broadcast axis value
      V_OP_TILING_CHECK((single_broadcast_axis < (int32_t)max_dim_len),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of broadcast axis %d is illegal",
                                                        single_broadcast_axis),
                        return false);
      broadcast_axis_ori[i] = single_broadcast_axis;
    }
    broadcast_axis_ori.resize(ori_broadcast_axis_len);
  } else {
    broadcast_axis_ori.clear();
  }

  // init disable fuse axes
  std::size_t ori_disable_fuse_axes_len = compileInfo.ori_disable_fuse_axes.size();
  for (std::size_t i = 0; i < ori_disable_fuse_axes_len; ++i) {
    int32_t single_disable_fuse_axes = compileInfo.ori_disable_fuse_axes[i];
    // convert axis index from negative to positive
    if (single_disable_fuse_axes < 0) {
      single_disable_fuse_axes = max_dim_len + single_disable_fuse_axes;
    }
    // check disable fuse axes value
    V_OP_TILING_CHECK((single_disable_fuse_axes < (int32_t)max_dim_len),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of disable fuse axes %d is illegal",
                                                      single_disable_fuse_axes),
                      return false);
    disable_fuse_axes_ori[i] = single_disable_fuse_axes;
  }
  disable_fuse_axes_ori.resize(ori_disable_fuse_axes_len);

  std::sort(reduce_axis_ori.begin(), reduce_axis_ori.end());
  std::sort(broadcast_axis_ori.begin(), broadcast_axis_ori.end());

  block_size = compileInfo.min_block_size;

  return true;
}

NormBroadcastMode Norm::JudgeBroadcastMode(const std::array<int64_t, NORM_MAX_DIM_LEN>& before_broadcast_shape) {
  bool is_no_broadcast = true;
  bool is_same_reduce = true;
  bool is_opposite_reduce = true;
  bool is_all_broadcast = true;

  std::size_t ori_reduce_axis_len = reduce_axis_ori.size();
  std::size_t count_reduce_axis = 0;

  for (std::size_t i = 0; i < max_dim_len; i++) {
    if (input_shape_ori[i] == before_broadcast_shape[i] && input_shape_ori[i] != 1) {
      // unbroadcast axis
      is_all_broadcast = false;
      if (count_reduce_axis != ori_reduce_axis_len && (int32_t)i == reduce_axis_ori[count_reduce_axis]) {
        // reduce axis
        count_reduce_axis++;
        is_same_reduce = false;
      } else {
        // unreduce axis
        is_opposite_reduce = false;
      }
    } else {
      // broadcast axis
      is_no_broadcast = false;
      if (count_reduce_axis != ori_reduce_axis_len && (int32_t)i == reduce_axis_ori[count_reduce_axis]) {
        // reduce axis
        count_reduce_axis++;
        is_opposite_reduce = false;
      } else {
        // unreduce axis
        is_same_reduce = false;
      }
    }
  }

  if (is_no_broadcast) {
    return NO_BROADCAST;
  }

  if (is_same_reduce) {
    return SAME_REDUCE;
  }

  if (is_opposite_reduce) {
    return OPPOSITE_REDUCE;
  }

  if (is_all_broadcast) {
    return ALL_BROADCAST;
  }

  return OTHERS;
}

int32_t Norm::CalcMinEliminateOneIndex() {
  constexpr int32_t loop_location = 2;
  for (int32_t i = (int32_t)max_dim_len - loop_location; i >= 0; i--) {
    if (IsElementInVector(reduce_axis_ori, i) != IsElementInVector(reduce_axis_ori, i + 1)) {
      return i + 1;
    }
    if (compileInfo.is_broadcast_axis_known) {
      if (IsElementInVector(broadcast_axis_ori, i) != IsElementInVector(broadcast_axis_ori, i + 1)) {
        return i + 1;
      }
    }
  }

  return 0;
}

bool Norm::EliminateOne() {
  // no fuse case
  bool is_no_fuse_axis = !compileInfo.is_fuse_axis;
  // dont have after broadcast input
  bool is_no_after_broadcast_axis = before_broadcast_input_num == compileInfo.input_type.size();
  // have before broadcast input but unify broadcast axis is unknown
  bool is_unify_broadcast_axis_unknown = before_broadcast_input_num > 0 && !compileInfo.is_broadcast_axis_known;
  // disable fuse axes
  bool is_disable_fuse_axes = !disable_fuse_axes_ori.empty();

  bool is_cannot_eliminate_one = is_no_fuse_axis || is_no_after_broadcast_axis ||
                                 is_unify_broadcast_axis_unknown || is_disable_fuse_axes;
  if (is_cannot_eliminate_one) {
    return true;
  }

  // back is not 1
  if (input_shape_ori.back() != 1) {
    return true;
  }
  int32_t min_index = CalcMinEliminateOneIndex();
  for (int32_t i = (int32_t)max_dim_len - 1; i >= min_index; i--) {
    // last dim is one
    if (input_shape_ori[i] == 1) {
      if (reduce_axis_ori.empty()) {
        break;
      }
      if (compileInfo.is_broadcast_axis_known && broadcast_axis_ori.empty()) {
        break;
      }
      input_shape_ori.pop_back();
      if (reduce_axis_ori.back() == i) {
        reduce_axis_ori.pop_back();
      }
      if (compileInfo.is_broadcast_axis_known && broadcast_axis_ori.back() == i) {
        broadcast_axis_ori.pop_back();
      }
      continue;
    }
    break;
  }

  // after remove 1, reduce axis is empty
  // input shape (A, ) -> (1, A)
  // reduce axis () -> (0, )
  // broadcast axis (m, ) -> (0, m + 1)
  if (reduce_axis_ori.empty()) {
    input_shape_ori.insert(input_shape_ori.begin(), 1);
    reduce_axis_ori.emplace_back(0);
    for (auto& j : broadcast_axis_ori) {
      j++;
    }
    if (compileInfo.is_broadcast_axis_known) {
      broadcast_axis_ori.insert(broadcast_axis_ori.begin(), 0);
    }
  }

  return true;
}

bool Norm::FusedAxis() {
  // fuse axes of the same type
  // 0 means common axis
  // 1 means only reduce axis
  // 2 means only broadcast axis
  // 3 means broadcast and reduce axis
  int32_t common_flag = 0;
  int32_t reduce_flag = 1;
  int32_t broadcast_flag = 2;
  int32_t reduce_broadcast_flag = 3;
  int32_t modulo = 10;
  int32_t cnt = 1;

  std::array<int32_t, NORM_MAX_DIM_LEN> flags = {common_flag};
  for (const auto& idx : reduce_axis_ori) {
    flags[idx] = reduce_flag;
  }

  for (const auto& idx : broadcast_axis_ori) {
    if (flags[idx] == reduce_flag) {
      flags[idx] = reduce_broadcast_flag;
    } else {
      flags[idx] = broadcast_flag;
    }
  }

  for (const auto& idx: disable_fuse_axes_ori) {
    flags[idx] += modulo * (cnt++);
  }

  std::size_t first = 0;
  std::size_t second = 0;
  std::size_t length = input_shape_ori.size();

  std::size_t capacity_shape = 0;
  std::size_t capacity_reduce_axis = 0;
  std::size_t capacity_broadcast_axis = 0;

  while (second <= length) {
    if (second <= length - 1 && flags[first] == flags[second]) {
      // look for different type idx
      second++;
    } else {
      // fuse same type idx
      input_shape[capacity_shape] = std::accumulate(input_shape_ori.begin() + first, input_shape_ori.begin() + second,
                                                    1, std::multiplies<int64_t>());

      bool is_reduce_axis = (second <= length - 1 && flags[first] == reduce_flag) ||
                            (second == length && flags[second - 1] == reduce_flag);
      bool is_pad_reduce_axis = (second <= length - 1 && flags[first] % modulo == reduce_flag) ||
                                (second == length && flags[second - 1] % modulo == reduce_flag);
      bool is_broadcast_axis = (second <= length - 1 && flags[first] == broadcast_flag) ||
                               (second == length && flags[second - 1] == broadcast_flag);
      bool is_pad_broadcast_axis = (second <= length - 1 && flags[first] % modulo == broadcast_flag) ||
                                   (second == length && flags[second - 1] % modulo == broadcast_flag);
      bool is_reduce_broadcast_axis = (second <= length - 1 && flags[first] == reduce_broadcast_flag) ||
                                      (second == length && flags[second - 1] == reduce_broadcast_flag);
      bool is_pad_reduce_broadcast_axis = (second <= length - 1 && flags[first] % modulo == reduce_broadcast_flag) ||
                                          (second == length && flags[second - 1] % modulo == reduce_broadcast_flag);

      bool reduce_include = is_reduce_axis || is_reduce_broadcast_axis ||
                            is_pad_reduce_axis || is_pad_reduce_broadcast_axis;
      bool broadcast_include = is_broadcast_axis || is_reduce_broadcast_axis ||
                               is_pad_broadcast_axis || is_pad_reduce_broadcast_axis;
      if (reduce_include) {
        reduce_axis[capacity_reduce_axis] = capacity_shape;
        capacity_reduce_axis++;
      }
      if (broadcast_include) {
        broadcast_axis[capacity_broadcast_axis] = capacity_shape;
        capacity_broadcast_axis++;
      }
      capacity_shape++;
      first = second;
      second++;
    }
  }

  input_shape.resize(capacity_shape);
  reduce_axis.resize(capacity_reduce_axis);
  broadcast_axis.resize(capacity_broadcast_axis);

  return true;
}

bool Norm::GetNormPattern() {
  if (compileInfo.is_broadcast_axis_known) {
    broadcast_pattern = CalcPattern(input_shape, broadcast_axis);
  } else {
    for (std::size_t i = 0; i < before_broadcast_input_num; i++) {
      auto single_mode = JudgeBroadcastMode(before_broadcast_shapes[i]);
      int32_t weight = 10;
      broadcast_pattern = broadcast_pattern * weight + single_mode;
      // cannot fuse axis
      if (single_mode == OTHERS) {
        input_shape = input_shape_ori;
        reduce_axis = reduce_axis_ori;
      }
    }
  }

  reduce_pattern = CalcPattern(input_shape, reduce_axis);
  norm_pattern = reduce_pattern * NORM_REDUCE_PATTERN_WEIGHT + broadcast_pattern;

  return true;
}

bool Norm::GetUbSizeInfo() {
  try {
    const auto& available_ub_size_vec = compileInfo.available_ub_size.at(std::to_string(norm_pattern));
    std::size_t expect_vec_len = 3;
    V_OP_TILING_CHECK((available_ub_size_vec.size() == expect_vec_len),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "size of available_ub_size_vec is %zu that is illegal",
                                                      available_ub_size_vec.size()),
                      return false);
    std::size_t common_max_ub_count_index = 0;
    std::size_t workspace_max_ub_count_index = 1;
    std::size_t pad_max_ub_count_index = 2;
    common_max_ub_count = available_ub_size_vec[common_max_ub_count_index];
    workspace_max_ub_count = available_ub_size_vec[workspace_max_ub_count_index];
    pad_max_ub_count = available_ub_size_vec[pad_max_ub_count_index];
    V_OP_TILING_CHECK((common_max_ub_count > 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "common_max_ub_count is %d that is illegal",
                                                      common_max_ub_count),
                      return false);
    V_OP_TILING_CHECK((workspace_max_ub_count > 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "workspace_max_ub_count is %d that is illegal",
                                                      workspace_max_ub_count),
                      return false);
    V_OP_TILING_CHECK((pad_max_ub_count > 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "pad_max_ub_count is %d that is illegal",
                                                      pad_max_ub_count),
                      return false);
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get ub size info failed. Error message: %s", e.what());
    return false;
  }

  return true;
}

bool Norm::CalcInputAlignShape() {
  for (std::size_t i = 0; i < input_shape.size(); i++) {
    if (i == input_shape.size() - 1) {
      input_align_shape[i] = CeilDiv(input_shape[i], block_size) * block_size;
      continue;
    }
    input_align_shape[i] = input_shape[i];
  }
  input_align_shape.resize(input_shape.size());

  return true;
}

bool Norm::IsNeedPartialReorder() {
  if (reduce_axis.size() <= 1) {
    return false;
  }

  std::size_t discontinuous_axis_num = 1;
  for (std::size_t i = 1; i < reduce_axis.size(); i++) {
    // continuous reduce axis
    if (reduce_axis[i] == reduce_axis[i - 1] + 1) {
      continue;
    } else {
      discontinuous_axis_num++;
    }
  }
  constexpr std::size_t min_discontinuous_axis_num = 2;
  is_discontinuous_reduce_axis = discontinuous_axis_num >= min_discontinuous_axis_num;

  return is_discontinuous_reduce_axis && input_shape.back() < block_size;
}


bool Norm::IsNeedWorkspace() {
  // pure move
  if (!is_discontinuous_reduce_axis && reduce_product == 1) {
    return false;
  }

  int64_t shape_product = 1;
  for (std::size_t i = 0; i < input_align_shape.size(); i++) {
    if (IsElementInVector(reduce_axis, i)) {
      shape_product = input_align_shape[i] * shape_product;
    }
  }

  // nlast reduce need judge r_product * align(last_a) and ub_size
  for (std::size_t i = last_r_axis_index + 1; i < input_align_shape.size(); i++) {
    shape_product = shape_product * input_align_shape[i];
  }

  return shape_product > common_max_ub_count;
}

bool Norm::IsNeedAlignedInUb() {
  // data move should be continuous
  if (!is_continuous_data_move) {
    return false;
  }

  // last dim is not align
  if (input_shape.back() % block_size == 0) {
    return false;
  }

  // num of common axis should be less than or equal to 1
  if (input_shape.size() - reduce_axis.size() > 1) {
    return false;
  }

  // last dim of input_align_shape should be small enough
  if (pad_max_ub_count / compileInfo.pad_max_entire_size < input_align_shape.back()) {
    return false;
  }

  // num of not_1 axes should be greater than or equal to 2
  std::size_t count_one = 0;
  for (const auto& singe_dim : input_shape) {
    if (singe_dim == 1) {
      count_one++;
    }
  }
  std::size_t min_not_one_axis_num = 2;
  if (input_shape.size() - count_one < min_not_one_axis_num) {
    return false;
  }

  // reduce product should be less than pad_max_ub_count
  int64_t shape_product = 1;
  for (std::size_t i = 0; i < input_align_shape.size(); i++) {
    if (IsElementInVector(reduce_axis, i)) {
      shape_product = input_align_shape[i] * shape_product;
    }
  }

  return shape_product <= pad_max_ub_count;
}

bool Norm::GetPartialReorderBlockTilingInfo() {
  is_need_workspace = true;
  sch_type = NORM_PARTIAL_REORDER_SCH_TYPE;
  ub_size = workspace_max_ub_count;

  int64_t right_product = after_reduce_shape_product;
  // if after reduce shape product < block_size, block_dim is 1
  if (right_product < block_size) {
    tilingInfo.block_tiling_axis = first_a_axis_index;
    tilingInfo.block_tiling_factor = input_shape[first_a_axis_index];
    return true;
  }

  int64_t right_align_product = after_reduce_align_shape_product;
  for (std::size_t i = 0; i < input_shape.size(); i++) {
    // block split A
    if (IsElementInVector(reduce_axis, i)) {
      continue;
    }
    // only block split at first a axis can enable multi_core
    if (i == 0) {
      int64_t max_block_factor = ub_size / (right_align_product / input_align_shape[i]);
      int32_t start_core_num = compileInfo.core_num < input_shape[i] ? compileInfo.core_num : input_shape[i];
      for (int32_t tilling_core_num = start_core_num; tilling_core_num <= input_shape[i];
           tilling_core_num += compileInfo.core_num) {
        int64_t cur_min_block_factor = CeilDiv(input_shape[i], tilling_core_num);
        if (cur_min_block_factor > max_block_factor) {
          continue;
        }
        for (int32_t cur_block_dim = tilling_core_num; cur_block_dim >= 1; cur_block_dim--) {
          int64_t cur_block_factor = CeilDiv(input_shape[i], cur_block_dim);
          if (cur_block_factor > max_block_factor) {
            continue;
          }
          int64_t tail_block_factor = CalcRemainder(input_shape[i], cur_block_factor);
          int64_t tail_block_inner_elem_count = tail_block_factor * (right_product / input_shape[i]);
          if (tail_block_inner_elem_count < block_size) {
            tilingInfo.block_tiling_axis = (int32_t)i;
            tilingInfo.block_tiling_factor = cur_block_factor;
            return true;
          }
        }
      }
    }
    // ub should can put in A axis after block_inner
    if (right_align_product / input_align_shape[i] <= ub_size) {
      int64_t cur_block_factor = ub_size / (right_align_product / input_align_shape[i]);
      tilingInfo.block_tiling_axis = (int32_t)i;
      tilingInfo.block_tiling_factor = std::min(cur_block_factor, input_shape[i]);
      return true;
    }
    right_product = right_product / input_shape[i];
    right_align_product = right_align_product / input_align_shape[i];
  }

  return false;
}

bool Norm::JudgeCurDimSplitBlock(const int64_t& right_product, const std::size_t& index) {
  int64_t cur_block_factor = CeilDiv(block_size, right_product / input_shape[index]);
  if (cur_block_factor >= input_shape[index]) {
    tilingInfo.block_tiling_axis = index;
    tilingInfo.block_tiling_factor = input_shape[index];
    return true;
  }

  for (; cur_block_factor <= input_shape[index]; cur_block_factor++) {
    // elements in tail core > block can not be considered in continuous_data_move case
    bool is_tail_considered_as_entire = (input_shape[index] % cur_block_factor == 0) || is_continuous_data_move;
    int64_t tail_block_factor =
      is_tail_considered_as_entire ? cur_block_factor : input_shape[index] % cur_block_factor;
    int64_t tail_block_inner_elem_count = tail_block_factor * (right_product / input_shape[index]);
    if (tail_block_inner_elem_count >= block_size) {
      tilingInfo.block_tiling_axis = index;
      tilingInfo.block_tiling_factor = cur_block_factor;
      return true;
    }
  }

  return false;
}

bool Norm::JudgeCurDimSplitBlock(const int64_t& left_product, const int64_t& right_product,
                                 const std::size_t& index, const int64_t& max_block_factor) {
  for (int32_t tilling_core_num = compileInfo.core_num; tilling_core_num <= left_product * input_shape[index];
       tilling_core_num += compileInfo.core_num) {
    if (left_product > tilling_core_num) {
      continue;
    }
    int64_t cur_block_dim = tilling_core_num / left_product;
    // only entire core should be considered in continuous_data_move case
    int64_t considered_block_dim = (cur_block_dim != 1 && is_continuous_data_move) ?
                                   cur_block_dim - 1 : cur_block_dim;
    int64_t cur_block_factor = CeilDiv(input_shape[index], considered_block_dim);
    // max_block_factor has default value 0
    if (max_block_factor != 0 && cur_block_factor > max_block_factor) {
      continue;
    }
    for (; cur_block_dim >= 1; cur_block_dim--) {
      if (cur_block_dim != tilling_core_num / left_product) {
        considered_block_dim = (cur_block_dim != 1 && is_continuous_data_move) ?
                                cur_block_dim - 1 : cur_block_dim;
        cur_block_factor = CeilDiv(input_shape[index], considered_block_dim);
      }
      if (max_block_factor != 0 && cur_block_factor > max_block_factor) {
        continue;
      }
      bool is_tail_considered_as_entire = (input_shape[index] % cur_block_factor == 0) || is_continuous_data_move;
      int64_t tail_block_factor =
        is_tail_considered_as_entire ? cur_block_factor : input_shape[index] % cur_block_factor;
      int64_t post_right_product = right_product / input_shape[index];
      int64_t tail_block_inner_elem_count = tail_block_factor * post_right_product;
      if (tail_block_inner_elem_count >= block_size) {
        // calculation of block factor should according to cur_block_dim,
        // but tail_block_inner_elem_count > block_size is according to considered_block_dim,
        // cur_block_dim >= considered_block_dim in continuous_data_move case,
        // so actual_block_factor may not large enough,
        // if actual_block_factor * (right_product / input_shape[index]) < block,
        // actual_block_factor should be refined
        int64_t actual_block_factor = CeilDiv(input_shape[index], cur_block_dim);
        tilingInfo.block_tiling_axis = index;
        tilingInfo.block_tiling_factor = is_continuous_data_move &&
                                         (actual_block_factor * post_right_product < block_size) ?
                                         CeilDiv(block_size, post_right_product) :
                                         actual_block_factor;

        return true;
      }
    }
  }

  return false;
}

bool Norm::GetWorkspaceBlockTilingInfo() {
  // left product: calc core num
  // right_product: mte3 element num >= block
  // right_align_product: ub can put in A axis after block_inner
  int64_t left_product = 1;
  int64_t right_align_product = after_reduce_align_shape_product;
  int64_t right_product = after_reduce_shape_product;
  bool exist_after_reduce_tensor = compileInfo.exist_output_after_reduce || compileInfo.exist_workspace_after_reduce;
  right_product = exist_after_reduce_tensor ? right_product : right_product * reduce_product;

  // if right_product < block_size, block_dim = 1
  if (right_product < block_size) {
    tilingInfo.block_tiling_axis = first_a_axis_index;
    tilingInfo.block_tiling_factor = input_shape[first_a_axis_index];
    return true;
  }

  std::size_t a_axis_count = 0;
  for (std::size_t i = 0; i < input_shape.size(); i++) {
    // block split A
    if (IsElementInVector(reduce_axis, i)) {
      right_product = exist_after_reduce_tensor ? right_product : right_product / input_shape[i];
      continue;
    }
    a_axis_count++;
    int64_t post_left_product = left_product * input_shape[i];
    int64_t post_right_product = right_product / input_shape[i];
    int64_t post_right_align_product = right_align_product / input_align_shape[i];

    // the A after reduce can not put in ub
    if (post_right_align_product > ub_size) {
      left_product = post_left_product;
      right_product = post_right_product;
      right_align_product = post_right_align_product;
      continue;
    }

    if (post_right_product <= block_size && post_left_product < compileInfo.core_num) {
      if (JudgeCurDimSplitBlock(right_product, i)) {
        return true;
      }
    } else if (post_right_product <= block_size || post_left_product >= compileInfo.core_num) {
      // max_block_factor: ub can put in A axis after block_inner
      int64_t max_block_factor = ub_size / post_right_align_product;
      if (JudgeCurDimSplitBlock(left_product, right_product, i, max_block_factor)) {
        return true;
      }
    }

    // all right_product > block_size and left_product < compileInfo.core_num
    // split last a and factor is 1
    if (a_axis_count == input_shape.size() - reduce_axis.size()) {
      tilingInfo.block_tiling_axis = i;
      tilingInfo.block_tiling_factor = 1;
      return true;
    }

    left_product = post_left_product;
    right_product = post_right_product;
    right_align_product = post_right_align_product;
  }

  return false;
}

bool Norm::GetBlockTilingInfo() {
  if (is_partial_reorder) {
    bool ret = GetPartialReorderBlockTilingInfo();
    // partial reorder sch enables multi_core if and only if the block_tiling_axis is 0
    if (tilingInfo.block_tiling_axis == 0) {
      tilingInfo.block_dim = GetBlockDim(tilingInfo.block_tiling_axis, tilingInfo.block_tiling_factor);
    } else {
      tilingInfo.block_dim = 1;
    }
    return ret;
  }

  if (is_need_workspace) {
    return GetWorkspaceBlockTilingInfo();
  }

  // left product: calc core num
  // right_product: mte3 element num >= block
  int64_t left_product = 1;
  int64_t right_product = after_reduce_shape_product;
  right_product = compileInfo.exist_output_after_reduce ? right_product : right_product * reduce_product;

  // if after shape product < block_size, block_dim = 1
  if (right_product < block_size) {
    tilingInfo.block_tiling_axis = first_a_axis_index;
    tilingInfo.block_tiling_factor = input_shape[first_a_axis_index];
    return true;
  }

  std::size_t a_axis_count = 0;
  for (std::size_t i = 0; i < input_shape.size(); i++) {
    // block split A
    if (IsElementInVector(reduce_axis, i)) {
      right_product = compileInfo.exist_output_after_reduce ? right_product : right_product / input_shape[i];
      continue;
    }
    a_axis_count++;
    int64_t post_left_product = left_product * input_shape[i];
    int64_t post_right_product = right_product / input_shape[i];

    if (post_right_product <= block_size && post_left_product < compileInfo.core_num) {
      if (JudgeCurDimSplitBlock(right_product, i)) {
        return true;
      }
    } else if (post_right_product <= block_size || post_left_product >= compileInfo.core_num) {
      if (JudgeCurDimSplitBlock(left_product, right_product, i)) {
        return true;
      }
    }
    // all right_product > block_size and left_product < compileInfo.core_num
    // split last a and factor is 1
    if (a_axis_count == input_shape.size() - reduce_axis.size()) {
      tilingInfo.block_tiling_axis = i;
      tilingInfo.block_tiling_factor = 1;
      return true;
    }
    left_product = post_left_product;
    right_product = post_right_product;
  }

  return false;
}

int32_t Norm::GetBlockDim(int32_t tiling_axis, int64_t tiling_factor) {
  int32_t block_dim = 1;
  for (int32_t i = 0; i <= tiling_axis; i++) {
    if (IsElementInVector(reduce_axis, i)) {
      continue;
    }
    if (i == tiling_axis) {
      if (tiling_factor != 0) {
        block_dim = (int32_t)CeilDiv(input_shape[i], tiling_factor) * block_dim;
      }
    } else {
      block_dim = (int32_t)input_shape[i] * block_dim;
    }
  }
  return block_dim;
}

bool Norm::ProcessReorderAxis() {
  /* InputShape: a0,r0,a1,r1,a2,r2,r3,a3
   *                    |---> block_tiling_axis
   *                    |---> core = a0*a1
   *                    |---> fused_block_tiling_axis= a0
   * ReorderShape: |a0,a1,a2|r0,r1,r2,r3|a3
   *                                   |---> last_r_axis_index
   * */
  std::array<bool, NORM_MAX_DIM_LEN> reduce_flag = {false};

  reorderInfo.reorder_input_shape.resize(input_shape.size());
  reorderInfo.reorderPos_oriPos.resize(input_shape.size());

  std::size_t num_a = input_shape.size() - reduce_axis.size();
  for (const auto& item : reduce_axis) {
    reduce_flag[item] = true;
  }
  // position of first R index in reorder shape
  std::size_t pos_r = num_a - (input_shape.size() - (last_r_axis_index + 1));
  std::size_t pos_a = 0;

  // [0: last_r_axis_index]
  for (int32_t i = 0; i <= last_r_axis_index; i++) {
    if (reduce_flag[i]) {
      reorderInfo.reorder_input_shape[pos_r] = input_shape[i];
      reorderInfo.reorderPos_oriPos[pos_r] = i;
      pos_r++;
    } else {
      if (i < tilingInfo.block_tiling_axis) {
        reorderInfo.fused_block_tiling_axis.emplace_back(pos_a);
      }
      reorderInfo.reorder_input_shape[pos_a] = input_shape[i];
      reorderInfo.reorderPos_oriPos[pos_a] = i;
      pos_a++;
    }
  }

  // order last normal axis, maybe several axis
  for (int32_t i = last_r_axis_index + 1; i < (int32_t)input_shape.size(); i++) {
    if (i < tilingInfo.block_tiling_axis) {
      reorderInfo.fused_block_tiling_axis.emplace_back(pos_r);
    }
    reorderInfo.reorder_input_shape[pos_r] = input_shape[i];
    reorderInfo.reorderPos_oriPos[pos_r] = i;
    pos_r++;
  }

  for (int32_t i = 0; i < (int32_t)reorderInfo.reorderPos_oriPos.size(); i++) {
    if (reorderInfo.reorderPos_oriPos[i] == tilingInfo.block_tiling_axis) {
      block_tiling_axis_index_in_reorder = i;
      break;
    }
  }

  return true;
}

int64_t Norm::CalcReorderShapeProduct(int32_t axis_index, bool exclude_reduce_axis) {
  int64_t result = 1;
  for (int32_t i = axis_index + 1; i < (int32_t)reorderInfo.reorder_input_shape.size(); i++) {
    if (exclude_reduce_axis) {
      int32_t first_r_axis_in_reorder = last_r_axis_index - reduce_axis.size() + 1;
      int32_t last_r_axis_in_reorder = last_r_axis_index;
      if (first_r_axis_in_reorder <= i && i <= last_r_axis_in_reorder) {
        continue;
      }
    }
    if (IsElementInVector(reorderInfo.fused_block_tiling_axis, i)) {
      continue;
    }
    if (i == block_tiling_axis_index_in_reorder) {
      result = result * tilingInfo.block_tiling_factor;
    } else {
      result = result * reorderInfo.reorder_input_shape[i];
    }
  }

  return result;
}

int64_t Norm::CalcReorderAlignShapeProduct(int32_t axis_index) {
  int64_t result = 1;
  for (int32_t i = axis_index + 1; i < (int32_t)reorderInfo.reorder_input_shape.size(); i++) {
    if (IsElementInVector(reorderInfo.fused_block_tiling_axis, i)) {
      continue;
    }

    if (i == block_tiling_axis_index_in_reorder) {
      if (i == (int32_t)reorderInfo.reorder_input_shape.size() - 1) {
        result = result * (CeilDiv(tilingInfo.block_tiling_factor, block_size) * block_size);
      } else {
        result = result * tilingInfo.block_tiling_factor;
      }
    } else {
      if (i == (int32_t)reorderInfo.reorder_input_shape.size() - 1) {
        result = result * (CeilDiv(reorderInfo.reorder_input_shape[i], block_size) * block_size);
      } else {
        result = result * reorderInfo.reorder_input_shape[i];
      }
    }
  }
  // nlast reduce, axis is last A
  if (axis_index > last_r_axis_index) {
    result = result * reduce_product;
  }

  return result;
}

bool Norm::GetPartialReorderUbTilingInfo() {
  for (std::size_t i = 0; i < input_shape.size(); i++) {
    if (!IsElementInVector(reduce_axis, i)) {
      continue;
    }
    int64_t right_product_align = (i == input_shape.size() - 1 ?
                                   1:
                                   std::accumulate(input_align_shape.begin() + i + 1, input_align_shape.end(), 1,
                                                   std::multiplies<int64_t>()));
    if (right_product_align <= ub_size) {
      int64_t cur_ub_factor = ub_size / right_product_align;
      tilingInfo.ub_tiling_axis = (int32_t)i;
      tilingInfo.ub_tiling_factor = std::min(cur_ub_factor, input_shape[i]);
      return true;
    }
  }

  return false;
}

bool Norm::JudgeNormalCurDimSplitUb(const std::size_t& index) {
  // right_product: mte3 element num >= block
  // right_align_product: ub can put in A axis after block_inner
  int64_t right_product = CalcReorderShapeProduct(index, compileInfo.exist_output_after_reduce);
  int64_t right_product_align = CalcReorderAlignShapeProduct(index);
  if (right_product_align > ub_size) {
    return false;
  }

  int64_t current_dim = ((int32_t)index == block_tiling_axis_index_in_reorder) ?
                        tilingInfo.block_tiling_factor:
                        reorderInfo.reorder_input_shape[index];
  int64_t current_dim_tail = ((int32_t)index == block_tiling_axis_index_in_reorder) ?
                             reorderInfo.reorder_input_shape[index] % tilingInfo.block_tiling_factor:
                             reorderInfo.reorder_input_shape[index];
  // index is last a, some cases can early stop
  bool is_early_stop_case = (int32_t)index == last_a_axis_index_in_reorder &&
                            current_dim * right_product < block_size;
  if (is_early_stop_case) {
    // (15, 10000) and have after reduce output
    // ub can't split except that block dim = 1 and is_continuous_data_move
    bool early_stop_return_false = current_dim * right_product_align > ub_size &&
                                   !(is_continuous_data_move && tilingInfo.block_dim == 1);
    // current_dim * right_product_align <= ub_size
    // block dim = 1
    // ub can split this dim
    bool early_stop_return_true = current_dim * right_product_align <= ub_size && tilingInfo.block_dim == 1;
    if (early_stop_return_false) {
      return false;
    } else if (early_stop_return_true) {
      tilingInfo.ub_tiling_axis = index;
      tilingInfo.ub_tiling_factor = current_dim;
      return true;
    }
  }
  if (is_continuous_data_move) {
    current_dim_tail = current_dim;
  }
  int64_t cur_ub_factor = ub_size / right_product_align;
  for (; cur_ub_factor >= 1; cur_ub_factor--) {
    int64_t tail_ub_inner_elem_count = block_size;
    // the tail of entire core can not be considered except that the tail of tail core is zero
    if (current_dim_tail == 0) {
      int64_t tail_ub_factor = CalcRemainder(current_dim, cur_ub_factor);
      tail_ub_inner_elem_count = tail_ub_factor * right_product;
    }
    int64_t tail_tail_ub_factor = CalcRemainder(current_dim_tail, cur_ub_factor);
    if (is_continuous_data_move) {
      tail_tail_ub_factor = cur_ub_factor;
    }
    int64_t tail_tail_ub_inner_elem_count = tail_tail_ub_factor * right_product;
    bool is_find = (tail_ub_inner_elem_count >= block_size && tail_tail_ub_inner_elem_count >= block_size) ||
                   (is_continuous_data_move && tilingInfo.block_dim == 1);
    if (is_find) {
      tilingInfo.ub_tiling_axis = reorderInfo.reorderPos_oriPos[index];
      // entire_cut: ub factor tail should better equal to ub factor
      int64_t ub_loop = cur_ub_factor > current_dim ? 1 : CeilDiv(current_dim, cur_ub_factor);
      tilingInfo.ub_tiling_factor = CeilDiv(current_dim, ub_loop);
      // if entire_cut ub_factor is illegal, turn to common_cut ub_factor
      bool is_entire_cut_illegal = tilingInfo.ub_tiling_factor > ub_size / right_product_align ||
                                   tilingInfo.ub_tiling_factor * right_product < block_size;
      if (is_entire_cut_illegal) {
        tilingInfo.ub_tiling_factor = std::min(cur_ub_factor, current_dim);
      }
      // storage align last A when is nlast reduce, but align_factor * right_product is larger than max_ub
      bool is_common_cut_illegal = !is_last_axis_reduce && index == reorderInfo.reorder_input_shape.size() - 1 &&
                                   CeilDiv(tilingInfo.ub_tiling_factor, block_size) * block_size *
                                   right_product_align > ub_size;
      if (!is_common_cut_illegal) {
        return true;
      }
    }
  }

  return false;
}

bool Norm::JudgeWorkspaceCurDimSplitUb(const std::size_t& index) {
  // right_product: mte3 element num >= block
  // right_align_product: ub can put in A axis after block_inner
  // after reduce output compute at block_outer, so only consider mte3 of before reduce output
  int64_t right_product = CalcReorderShapeProduct(index, false);
  int64_t right_product_align = CalcReorderAlignShapeProduct(index);
  if (right_product_align > ub_size) {
    return false;
  }

  int64_t current_dim = reorderInfo.reorder_input_shape[index];
  int64_t cur_ub_factor = ub_size / right_product_align;
  for (; cur_ub_factor >= 1; cur_ub_factor--) {
    int64_t tail_ub_factor = CalcRemainder(current_dim, cur_ub_factor);
    int64_t tail_ub_tilling_inner_ddr_count = tail_ub_factor * right_product;
    if (tail_ub_tilling_inner_ddr_count >= block_size) {
      tilingInfo.ub_tiling_axis = reorderInfo.reorderPos_oriPos[index];
      // entire_cut: ub factor tail should better equal to ub factor
      int64_t ub_loop = cur_ub_factor > current_dim ? 1 : CeilDiv(current_dim, cur_ub_factor);
      tilingInfo.ub_tiling_factor = CeilDiv(current_dim, ub_loop);
      // if entire_cut ub_factor is illegal, turn to common_cut ub_factor
      bool is_entire_cut_illegal = tilingInfo.ub_tiling_factor > ub_size / right_product_align ||
                                   tilingInfo.ub_tiling_factor * right_product < block_size;
      if (is_entire_cut_illegal) {
        tilingInfo.ub_tiling_factor = std::min(cur_ub_factor, current_dim);
      }
      // storage align last R when is last reduce, but align_factor * right_product is larger than max_ub
      bool is_common_cut_illegal = is_last_axis_reduce && index == reorderInfo.reorder_input_shape.size() - 1 &&
                                   CeilDiv(tilingInfo.ub_tiling_factor, block_size) * block_size *
                                   right_product_align > ub_size;
      if (!is_common_cut_illegal) {
        return true;
      }
    }
  }

  return false;
}

bool Norm::GetUbTilingInfo() {
  if (is_partial_reorder) {
    return GetPartialReorderUbTilingInfo();
  }

  int32_t first_r_axis_in_reorder = last_r_axis_index - reduce_axis.size() + 1;
  int32_t last_r_axis_in_reorder = last_r_axis_index;

  // normal, ub split A
  if (!is_need_workspace) {
    // dont split ub if all reduce
    if (reduce_axis.size() == input_shape.size()) {
      return true;
    }
    for (int32_t i = 0; i < (int32_t)reorderInfo.reorder_input_shape.size(); i++) {
      if (IsElementInVector(reorderInfo.fused_block_tiling_axis, i)) {
        continue;
      }
      if (first_r_axis_in_reorder <= i && i <= last_r_axis_in_reorder) {
        continue;
      }
      if (JudgeNormalCurDimSplitUb(i)) {
        return true;
      }
    }
  } else {
    // workspace, ub split R
    for (int32_t i = 0; i < (int32_t)reorderInfo.reorder_input_shape.size(); i++) {
      if (IsElementInVector(reorderInfo.fused_block_tiling_axis, i)) {
        continue;
      }
      if (!(first_r_axis_in_reorder <= i && i <= last_r_axis_in_reorder)) {
        continue;
      }
      if (JudgeWorkspaceCurDimSplitUb(i)) {
        return true;
      }
    }
  }

  return false;
}

bool Norm::NeedRefineBlockTiling() {
  // block split last common axis, block factor can refine to align

  if (is_partial_reorder) {
    return false;
  }

  if (is_last_axis_reduce || tilingInfo.block_tiling_axis != (int32_t)input_shape.size() - 1) {
    return false;
  }
  int64_t refined_block_tiling_factor = CeilDiv(tilingInfo.block_tiling_factor, block_size) * block_size;
  if (refined_block_tiling_factor > input_shape[tilingInfo.block_tiling_axis]) {
    return false;
  }

  if (is_need_workspace) {
    if (refined_block_tiling_factor > ub_size) {
      return false;
    }
  }
  int64_t tail_dim = CalcRemainder(input_shape[tilingInfo.block_tiling_axis], refined_block_tiling_factor);

  return tail_dim >= block_size;
}

bool Norm::ProcessTiling()
{
  bool ret = true;
  // has A axis
  if (reduce_axis.size() != input_shape.size()) {
    ret = ret && GetBlockTilingInfo();
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetBlockTilingInfo error");
      return false;
    }
    if (NeedRefineBlockTiling()) {
      tilingInfo.block_tiling_factor = CeilDiv(tilingInfo.block_tiling_factor, block_size) * block_size;
    }
    if (tilingInfo.block_dim == -1) {
      tilingInfo.block_dim = GetBlockDim(tilingInfo.block_tiling_axis, tilingInfo.block_tiling_factor);
    }
  } else {
    tilingInfo.block_dim = 1;
  }
  ret = ret && ProcessReorderAxis();
  ret = ret && GetUbTilingInfo();
  if (!ret) {
    if (is_need_workspace) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetUbTilingInfo error");
      return false;
    }
    ret = true;
    is_need_workspace = true;
    sch_type = NORM_COMMON_SCH_TYPE;
    ub_size = workspace_max_ub_count;
    if (reduce_axis.size() != input_shape.size()) {
      ret = ret && GetBlockTilingInfo();
      if (!ret) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetBlockTilingInfo error");
        return false;
      }
      if (NeedRefineBlockTiling()) {
        tilingInfo.block_tiling_factor = CeilDiv(tilingInfo.block_tiling_factor, block_size) * block_size;
      }
      if (tilingInfo.block_dim == -1) {
        tilingInfo.block_dim = GetBlockDim(tilingInfo.block_tiling_axis, tilingInfo.block_tiling_factor);
      }
    } else {
      tilingInfo.block_dim = 1;
    }
    ret = ret && ProcessReorderAxis();
    ret = ret && GetUbTilingInfo();
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetUbTilingInfo error");
      return false;
    }
  }

  return true;
}

bool Norm::CalcTilingKey() {
  int64_t db_weight = 2000000000;
  int64_t is_broadcast_axis_known_weight = 1000000000;
  int64_t sch_type_weight = 100000000;
  int64_t norm_pattern_weight = 100;
  int64_t block_axis_weight = 10;

  tiling_key = db * db_weight + compileInfo.is_broadcast_axis_known * is_broadcast_axis_known_weight +
               sch_type * sch_type_weight + norm_pattern * norm_pattern_weight;

  int32_t block_key = (tilingInfo.block_tiling_axis == -1) ? NORM_NONE_SPLIT_KEY : tilingInfo.block_tiling_axis;
  tiling_key += block_key * block_axis_weight;
  int32_t ub_key = (tilingInfo.ub_tiling_axis == -1) ? NORM_NONE_SPLIT_KEY : tilingInfo.ub_tiling_axis;
  tiling_key += ub_key;

  tiling_key_str = std::to_string(tiling_key);

  return true;
}

bool Norm::CalcWorkspace() {
  std::size_t workspace_count = 0;
  if (!compileInfo.is_const || compileInfo.is_const_post) {
    try {
      const auto& workspace_info = compileInfo.workspace_info.at(tiling_key_str);
      workspace_count = workspace_info.size();
      if (workspace_count != 0) {
        int64_t shape_after_reduce_align_product = CalcAfterReduceShapeProduct(input_align_shape, reduce_axis);
        int64_t shape_before_reduce_align_product = 1;
        for (const auto& single_dim : input_align_shape) {
          shape_before_reduce_align_product *= single_dim;
        }

        for (std::size_t i = 0; i < workspace_count; i++) {
          // fake workspace is represented by 32
          // after reduce workspace tensor is represented by a negative number
          // before reduce workspace tensor is represented by a positive number
          if (workspace_info[i] == NORM_FAKE_WORKSPACE_SIZE) {
            workspace[i] = NORM_FAKE_WORKSPACE_SIZE;
          } else if (workspace_info[i] < 0) {
            workspace[i] = shape_after_reduce_align_product * workspace_info[i] * -1;
          } else {
            workspace[i] = shape_before_reduce_align_product * workspace_info[i];
          }
        }
      }
    } catch (const std::exception &e) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get workspace info error. Error message: %s", e.what());
      return false;
    }
  }
  workspace.resize(workspace_count);

  return true;
}

bool Norm::GetVarValue() {
  std::size_t count_var = 0;
  // obtain var value
  if (!compileInfo.is_const) {
    try {
      int32_t ub_tiling_factor_encode = 40000;
      int32_t block_tiling_factor_encode = 30000;
      int32_t known_broadcast_encode = 20000;
      int32_t unknown_broadcast_encode = 10000;
      int32_t dim_encode = 100;
      const auto& var_pattern = compileInfo.norm_vars.at(tiling_key_str);
      for (const auto& var : var_pattern) {
        if (var >= ub_tiling_factor_encode) {
          var_value[count_var] = (int32_t)tilingInfo.ub_tiling_factor;
          count_var++;
        } else if (var >= block_tiling_factor_encode) {
          var_value[count_var] = (int32_t)tilingInfo.block_tiling_factor;
          count_var++;
        } else if (var >= known_broadcast_encode) {
          var_value[count_var] = (int32_t)input_shape[var % unknown_broadcast_encode];
          count_var++;
        } else {
          var_value[count_var] = (int32_t)before_broadcast_shapes[
            (var % unknown_broadcast_encode) / dim_encode][(var % unknown_broadcast_encode) % dim_encode];
          count_var++;
        }
      }
      var_value.resize(count_var);
    } catch (const std::exception &e) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get var value error. Error message: %s", e.what());
      return false;
    }
  }
  return true;
}

bool Norm::CalcTiling() {
  /* Situations of CalcTiling include:
     1. input(known):
        status of compile: do others except FusedAxis
        status of runtime: do WriteTilingData
     2. input(unknown):
        do all process
  */
  bool ret = Init();

  if (compileInfo.is_const) {
    // input(known)
    // invoking the tiling interface during runtime
    // is_const_post: "true"->runtime, "false"->compile
    if (compileInfo.is_const_post) {
      ret = ret && EliminateOne();
      ret = ret && FusedAxis();
      ret = ret && GetNormPattern();
      tiling_key = norm_pattern;
      tiling_key_str = std::to_string(tiling_key);
      ret = ret && CalcInputAlignShape();
      ret = ret && CalcWorkspace();
      return ret;
    } else {
      reduce_axis = reduce_axis_ori;
      broadcast_axis = broadcast_axis_ori;
      input_shape = input_shape_ori;
    }
  } else if (!compileInfo.is_fuse_axis) {
    reduce_axis = reduce_axis_ori;
    broadcast_axis = broadcast_axis_ori;
    input_shape = input_shape_ori;
  } else {
    // input(unknown)
    ret = ret && EliminateOne();
    ret = ret && FusedAxis();
  }

  ret = ret && GetNormPattern();
  ret = ret && GetUbSizeInfo();
  // calculate align shape
  ret = ret && CalcInputAlignShape();
  after_reduce_shape_product = CalcAfterReduceShapeProduct(input_shape, reduce_axis);
  after_reduce_align_shape_product = CalcAfterReduceShapeProduct(input_align_shape, reduce_axis);
  reduce_product = CalcReduceShapeProduct(input_shape, reduce_axis);

  for (int32_t i = 0; i < (int32_t)input_shape.size(); i++) {
    if (!IsElementInVector(reduce_axis, i)) {
      first_a_axis_index = i;
      break;
    }
  }
  last_r_axis_index = (int32_t)*max_element(reduce_axis.begin(), reduce_axis.end());
  is_last_axis_reduce = last_r_axis_index == (int32_t)input_shape.size() - 1;
  last_a_axis_index_in_reorder = is_last_axis_reduce ? (int32_t)input_shape.size() - (int32_t)reduce_axis.size() - 1 :
                                 (int32_t)input_shape.size() - 1;
  // calculate is_partial_reorder and is_discontinuous_reduce_axis
  is_partial_reorder = IsNeedPartialReorder();
  is_need_workspace = IsNeedWorkspace();
  is_continuous_data_move = is_last_axis_reduce && (!is_discontinuous_reduce_axis);
  is_align_and_remove_pad = IsNeedAlignedInUb();
  // determine ub size
  if (is_need_workspace) {
    ub_size = workspace_max_ub_count;
  } else if (is_align_and_remove_pad) {
    ub_size = pad_max_ub_count;
    sch_type = NORM_ALIGNED_IN_UB_SCH_TYPE;
  } else {
    ub_size = common_max_ub_count;
  }
  // calculate tiling info
  ret = ret && ProcessTiling();
  // calculate tiling key
  ret = ret && CalcTilingKey();
  // calculate workspace size
  ret = ret && CalcWorkspace();
  // get var value
  ret = ret && GetVarValue();

  return ret;
}

bool Norm::ConstInputProcPost() {
  // runtime
  try {
    run_info.SetBlockDim(compileInfo.const_block_dims.at(tiling_key_str));
    run_info.SetTilingKey(tiling_key);
    for (const auto& item : workspace) {
      run_info.AddWorkspace(item);
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ConstInputProcPost error. Error message: %s", e.what());
    return false;
  }

  return true;
}

bool Norm::WriteTilingData() {
  if (compileInfo.is_const_post) {
    // runtime
    return ConstInputProcPost();
  }

  if (compileInfo.is_const) {
    // compile
    // if split block
    if (tilingInfo.block_tiling_axis != -1) {
      run_info.AddTilingData((int32_t)tilingInfo.block_tiling_axis);
      run_info.AddTilingData((int32_t)tilingInfo.block_tiling_factor);
    }
    // if split ub
    if (tilingInfo.ub_tiling_axis != -1) {
      run_info.AddTilingData((int32_t)tilingInfo.ub_tiling_axis);
      run_info.AddTilingData((int32_t)tilingInfo.ub_tiling_factor);
    }
    run_info.SetBlockDim(tilingInfo.block_dim);
    return true;
  }

  run_info.SetTilingKey(tiling_key);
  run_info.SetBlockDim(tilingInfo.block_dim);

  for (const auto& item : workspace) {
    run_info.AddWorkspace(item);
  }

  for (const auto& item : var_value) {
    run_info.AddTilingData(item);
  }

  return true;
}

bool Norm::DoTiling() {
  bool ret = GetInput();
  ret = ret && CalcTiling();
  ret = ret && WriteTilingData();
  return ret;
}

bool NormCompileInfo::Check() {
  for (const auto& single_input_type : input_type) {
    V_OP_TILING_CHECK((single_input_type >= 0 && single_input_type <= (int32_t)NORM_MAX_INPUT_NUMS),
                      VECTOR_INNER_ERR_REPORT_TILIING(norm_op_type, "input type is %d that is illegal",
                                                      single_input_type),
                      return false);
  }
  V_OP_TILING_CHECK((core_num > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(norm_op_type, "core_num is %d that is illegal",
                                                    core_num),
                    return false);
  V_OP_TILING_CHECK((min_block_size > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(norm_op_type, "min_block_size is %d that is illegal",
                                                    min_block_size),
                    return false);
  V_OP_TILING_CHECK((pad_max_entire_size > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(norm_op_type, "pad_max_entire_size is %d that is illegal",
                                                    pad_max_entire_size),
                    return false);
  return true;
}

NormCompileInfo::NormCompileInfo(const std::string& op_type, const nlohmann::json& parsed_json_obj) {
  norm_op_type = op_type;
  OP_LOGD(norm_op_type.c_str(), "norm compile info construct func running");
  // reduce and broadcast axis
  ori_reduce_axis = parsed_json_obj.at("_ori_reduce_axis").get<std::vector<int32_t>>();
  if (parsed_json_obj.contains("_ori_broadcast_axis")) {
    ori_broadcast_axis = parsed_json_obj.at("_ori_broadcast_axis").get<std::vector<int32_t>>();
    is_broadcast_axis_known = true;
  }
  if (parsed_json_obj.contains("_disable_fuse_axes")) {
    ori_disable_fuse_axes = parsed_json_obj.at("_disable_fuse_axes").get<std::vector<int32_t>>();
  }
  // graph info
  input_type = parsed_json_obj.at("_input_type").get<std::vector<int32_t>>();
  exist_output_after_reduce = parsed_json_obj.at("_exist_output_after_reduce").get<bool>();
  exist_workspace_after_reduce = parsed_json_obj.at("_exist_workspace_after_reduce").get<bool>();
  available_ub_size =
    parsed_json_obj.at("_available_ub_size").get<std::unordered_map<std::string, std::vector<int32_t>>>();
  // common info
  const auto& common_info = parsed_json_obj.at("_common_info").get<std::vector<int32_t>>();
  std::size_t expect_common_info_len = 3;
  if (common_info.size() == expect_common_info_len) {
    std::size_t core_num_index = 0;
    std::size_t min_block_size_index = 1;
    std::size_t pad_max_entire_size_index = 2;
    core_num = common_info[core_num_index];
    min_block_size = common_info[min_block_size_index];
    pad_max_entire_size = common_info[pad_max_entire_size_index];
  }
  // workspace info
  if (parsed_json_obj.contains("_workspace_info")) {
    workspace_info =
      parsed_json_obj.at("_workspace_info").get<std::unordered_map<std::string, std::vector<int32_t>>>();
  }
  // fuse axis
  if (parsed_json_obj.contains("_fuse_axis")) {
    is_fuse_axis = parsed_json_obj.at("_fuse_axis").get<bool>();
  }
  // const
  if (parsed_json_obj.contains("_is_const")) {
    is_const = parsed_json_obj.at("_is_const").get<bool>();
  }
  if (is_const) {
    is_const_post = parsed_json_obj.at("_const_shape_post").get<bool>();
    if (is_const_post) {
      const_block_dims = parsed_json_obj.at("_const_block_dims").get<std::unordered_map<std::string, int32_t>>();
    }
  } else {
    norm_vars = parsed_json_obj.at("_norm_vars").get<std::unordered_map<std::string, std::vector<int32_t>>>();
  }
  check_success = Check();
}

bool NormTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "Norm tiling running");
  Norm norm(op_type, op_paras, norm_compile_info, run_info);
  return norm.DoTiling();
}

bool NormTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                 const OpInfo& op_info) const {
  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Norm custom tiling is not supported yet");
  return false;
}

std::shared_ptr<AutoTilingHandler> CreateNormTilingHandler(const std::string& op_type,
                                                           const std::string& pattern,
                                                           const nlohmann::json& parsed_compile_info) {
  auto compile_info_ptr = std::make_shared<NormTilingHandler>(op_type, pattern, parsed_compile_info);
  if (!compile_info_ptr->ParsedSuccess()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Norm parse compile info failed");
    return std::shared_ptr<AutoTilingHandler>(nullptr);
  }

  return compile_info_ptr;
}
}  // namespace optiling
