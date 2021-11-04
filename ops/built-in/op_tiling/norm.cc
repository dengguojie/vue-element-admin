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
 * \file norm.cc
 * \brief tiling function of op
 */
#include <algorithm>
#include <numeric>
#include "norm.h"
#include "error_log.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {

bool Norm::IsInVector(std::vector<int32_t>& shape, int32_t value)
{
  for (uint32_t i = 0; i < shape.size(); i++) {
    if (shape[i] == value) {
      return true;
    }
  }
  return false;
}

int64_t Norm::CalcAfterReduceShapeProduct(std::vector<int64_t>& shape, std::vector<int32_t>& axis)
{
  int64_t result = 1;
  for (std::size_t i = 0; i < shape.size(); i++) {
    if (!IsInVector(axis, i)) {
      result = result * shape[i];
    }
  }
  return result;
}

int64_t Norm::CalcReduceShapeProduct(std::vector<int64_t>& shape, std::vector<int32_t>& axis)
{
  int64_t result = 1;
  for (std::size_t i = 0; i < shape.size(); i++) {
    if (IsInVector(axis, i)) {
      result = result * shape[i];
    }
  }
  return result;
}

bool Norm::GetInput()
{
  int64_t max_product = -1;
  // find before reduce shape
  for (std::size_t i = 0; i < op_paras.GetInputsSize(); i++) {
    const auto& local_input_shape = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(i)->
            GetShape().GetDims();
    int64_t current_product = std::accumulate(local_input_shape.begin(), local_input_shape.end(), 1,
                                              std::multiplies<int64_t>());
    if (current_product > max_product) {
      max_product = current_product;
      for (std::size_t j = 0; j < local_input_shape.size(); j++) {
        input_shape_ori[j] = local_input_shape[j];
      }
      input_shape_ori.resize(local_input_shape.size());
    }
  }

  return true;
}

bool Norm::Init()
{
  try {
    const auto& local_reduce_axis = op_info.at("_ori_axis").get<std::vector<int32_t>>();
    for (std::size_t i = 0; i < local_reduce_axis.size(); i++) {
      reduce_axis_ori[i] = local_reduce_axis[i];
    }
    reduce_axis_ori.resize(local_reduce_axis.size());
    // Convert reduce axis (-1 -> length+1)
    // CHECK AXIS VALUE
    int32_t max_value = int32_t(input_shape_ori.size());
    int32_t min_value = -1 * max_value;
    for (std::size_t i = 0; i < reduce_axis_ori.size(); i++) {
      if (reduce_axis_ori[i] >= max_value || reduce_axis_ori[i] < min_value) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of axis is illegal.");
        return false;
      }
      if (reduce_axis_ori[i] < 0) {
        reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
      }
    }

    compileInfo.is_const = op_info.count("_reduce_shape_known") > 0 &&
                           op_info.at("_reduce_shape_known").get<bool>();
    compileInfo.is_fuse_axis = op_info.count("_fuse_axis") > 0 &&
                               op_info.at("_fuse_axis").get<bool>();
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Func of Init error. Error message: %s", e.what());
    return false;
  }

  return true;
}

bool Norm::FusedReduceAxis()
{
  /* fuse axes of the same type
   * if after fused, pattern is R, pattern will be AR by padding "1".
   * */
  std::vector<int32_t> pos(input_shape_ori.size());
  for (const auto& item : reduce_axis_ori) {
    pos[item] = 1;
  }

  // Fused serial axises which in same type.
  std::size_t first = 0;
  std::size_t second = 0;
  int64_t value_input = 0;
  int32_t value_axis = 0;
  std::size_t length = input_shape_ori.size();
  bool cond_0 = false;
  bool cond_1 = false;

  std::size_t capacity_shape = 0;
  std::size_t capacity_axis = 0;

  while (second <= length) {
    if (second <= length - 1 and pos[first] == pos[second]) {
      // look for unequal idx
      second += 1;
      continue;
    } else {
      // fused serial axises
      value_input = std::accumulate(input_shape_ori.begin() + first, input_shape_ori.begin() + second, 1,
                                    std::multiplies<int64_t>());
      input_shape[capacity_shape] = value_input;
      capacity_shape++;

      // cond_0: [first, second) is serial reduce_axises
      // cond_1: [first: ] is serial reduce_axises.
      cond_0 = second <= length - 1 and pos[second] == 0;
      cond_1 = second == length and pos[second - 1] == 1;
      if (cond_0 or cond_1) {
        value_axis = capacity_shape - 1;
        reduce_axis[capacity_axis] = value_axis;
        capacity_axis++;
      }
      first = second;
      second += 1;
      continue;
    }
  }

  input_shape.resize(capacity_shape);
  reduce_axis.resize(capacity_axis);

  // R -> A,R
  if (input_shape.size() == reduce_axis.size()) {
    input_shape.insert(input_shape.begin(), 1, 1);
    reduce_axis = {1};
  }
  return true;
}

bool Norm::GetCompileInfo()
{
  try {
    const auto& common_info = op_info.at("_common_info").get<std::vector<int32_t>>();
    V_CHECK_EQ(common_info.size(), 7,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "size of common_info should be 7"),
               return false);
    // Get Data
    compileInfo.core_num = common_info[0];
    compileInfo.min_block_size = common_info[1];
    compileInfo.is_keep_dims = (bool)common_info[2];
    compileInfo.max_ub_count = common_info[3];
    compileInfo.workspace_max_ub_count = common_info[4];
    compileInfo.pad_max_ub_count = common_info[5];
    compileInfo.pad_max_entire_size = common_info[6];
    // CHECK VALUE
    V_OP_TILING_CHECK(compileInfo.core_num > 0,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "core_num is %d that is illegal",
                                                      compileInfo.core_num),
                      return false);
    V_OP_TILING_CHECK(compileInfo.min_block_size > 0,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "min_block_size is %d that is illegal",
                                                      compileInfo.min_block_size),
                      return false);
    V_OP_TILING_CHECK(compileInfo.max_ub_count > 0,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "max_ub_count is %d that is illegal",
                                                      compileInfo.max_ub_count),
                      return false);
    V_OP_TILING_CHECK(compileInfo.workspace_max_ub_count > 0,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "workspace_max_ub_count is %d that is illegal",
                                                      compileInfo.workspace_max_ub_count),
                      return false);
    V_OP_TILING_CHECK(compileInfo.pad_max_ub_count > 0,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "pad_max_ub_count is %d that is illegal",
                                                      compileInfo.pad_max_ub_count),
                      return false);
    V_OP_TILING_CHECK(compileInfo.pad_max_entire_size > 0,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "pad_max_entire_size is %d that is illegal",
                                                      compileInfo.pad_max_entire_size),
                      return false);
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Func: GetCompileInfo error. Error message: %s", e.what());
    return false;
  }

  return true;
}

int32_t Norm::CalcPattern(std::vector<int64_t>& shape, std::vector<int32_t>& axis)
{
  int32_t pattern = 0;
  for (std::size_t i = 0; i < shape.size(); i++) {
    if (IsInVector(axis, i)) {
      pattern += 2 << (shape.size() - i - 1);
    } else {
      pattern += ((int)shape.size() - 2 - (int)i) >= 0 ? 2 << (shape.size() - 2 - i) : 1;
    }
  }
  return pattern;
}

bool Norm::IsNeedWorkspace()
{
  if (!is_split_block) {
    return true;
  }
  int64_t shape_product = 1;
  for (std::size_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      if (i == input_shape.size() - 1) {
        shape_product = (input_shape[i] + block_size - 1) / block_size * block_size * shape_product;
      } else {
        shape_product = input_shape[i] * shape_product;
      }
    }
  }
  // nlast reduce need judge r_product * align(last_a) and ub_size
  for (int32_t i = last_r_axis_index + 1; i < (int32_t)input_shape.size(); i++) {
    if (i == (int32_t)input_shape.size() - 1) {
      shape_product = shape_product * input_align_shape.back();
    } else {
      shape_product = shape_product * input_shape[i];
    }
  }

  return shape_product > compileInfo.max_ub_count ? true : false;
}

bool Norm::GetWorkspaceBlockTilingInfo()
{
  // reduce is workspace tensor and need after_reduce_size > ub_size
  int64_t left_product = 1;
  int64_t right_align_product = CalcAfterReduceShapeProduct(input_align_shape, reduce_axis);
  int64_t right_product = shape_after_reduce_product;
  // if after shape product < block_size, block_dim = 1
  if (right_product < block_size) {
    tilingInfo.block_tiling_axis = first_a_axis_index;
    tilingInfo.block_tiling_factor = input_shape[first_a_axis_index];
    return true;
  }

  for (std::size_t i = 0; i < input_shape.size(); i++) {
    // block split A
    if (IsInVector(reduce_axis, i)) {
      continue;
    }
    // the A after reduce can not put in ub
    if (right_align_product / input_align_shape[i] > ub_size) {
      left_product = left_product * input_shape[i];
      right_product = right_product / input_shape[i];
      right_align_product = right_align_product / input_align_shape[i];
      continue;
    }
    // max_block_tiling_factor: UB can store m * inner_ub_count
    int64_t max_block_tiling_factor = ub_size / (right_align_product / input_align_shape[i]);

    if (right_product / input_shape[i] <= block_size && left_product * input_shape[i] < compileInfo.core_num) {
      int64_t cur_block_factor =
          (block_size + right_product / input_shape[i] - 1) / (right_product / input_shape[i]);
      if (cur_block_factor <= input_shape[i]) {
        for (; cur_block_factor <= input_shape[i]; cur_block_factor++) {
          int64_t tail_block_factor =
              input_shape[i] % cur_block_factor == 0 ? cur_block_factor : input_shape[i] % cur_block_factor;
          int64_t tail_block_tilling_inner_ddr_count = tail_block_factor * (right_product / input_shape[i]);
          if (tail_block_tilling_inner_ddr_count >= block_size) {
            tilingInfo.block_tiling_axis = i;
            tilingInfo.block_tiling_factor = cur_block_factor;
            return true;
          }
        }
      } else {
        tilingInfo.block_tiling_axis = i;
        tilingInfo.block_tiling_factor = input_shape[i];
        return true;
      }
    } else if (right_product / input_shape[i] <= block_size || left_product * input_shape[i] >= compileInfo.core_num) {
      for (int32_t tilling_core_num = compileInfo.core_num; tilling_core_num <= left_product * input_shape[i];
           tilling_core_num += compileInfo.core_num) {
        if (left_product > tilling_core_num) {
          continue;
        }
        int64_t cur_block_dim = tilling_core_num / left_product;
        int64_t cur_block_factor = (input_shape[i] + cur_block_dim - 1) / cur_block_dim;
        if (cur_block_factor > max_block_tiling_factor) {
          continue;
        }
        int64_t tail_block_factor =
            input_shape[i] % cur_block_factor == 0 ? cur_block_factor : input_shape[i] % cur_block_factor;
        int64_t tail_block_tilling_inner_ddr_count = tail_block_factor * (right_product / input_shape[i]);
        if (tail_block_tilling_inner_ddr_count < block_size) {
          for (cur_block_dim = cur_block_dim - 1; cur_block_dim >= 1; cur_block_dim--) {
            cur_block_factor = (input_shape[i] + cur_block_dim - 1) / cur_block_dim;
            if (cur_block_factor > max_block_tiling_factor) {
              continue;
            }
            tail_block_factor =
                input_shape[i] % cur_block_factor == 0 ? cur_block_factor : input_shape[i] % cur_block_factor;
            tail_block_tilling_inner_ddr_count = tail_block_factor * (right_product / input_shape[i]);
            if (tail_block_tilling_inner_ddr_count >= block_size) {
              tilingInfo.block_tiling_axis = i;
              tilingInfo.block_tiling_factor = cur_block_factor;
              return true;
            }
          }
        } else {
          tilingInfo.block_tiling_axis = i;
          tilingInfo.block_tiling_factor = cur_block_factor;
          return true;
        }
      }
    }
    left_product = left_product * input_shape[i];
    right_product = right_product / input_shape[i];
    right_align_product = right_align_product / input_align_shape[i];
  }
  return false;
}

bool Norm::GetBlockTilingInfo()
{
  if (is_partial_reorder) {
    // don't split block
    tilingInfo.block_tiling_axis = first_a_axis_index;
    tilingInfo.block_tiling_factor = input_shape[first_a_axis_index];
    is_need_workspace = true;
    sch_type = 2;
    ub_size = compileInfo.workspace_max_ub_count;
    return true;
  }
  if (is_need_workspace) {
    return GetWorkspaceBlockTilingInfo();
  }
  int64_t right_product = shape_after_reduce_product;
  int64_t left_product = 1;
  // output of each block > 32B, normal sch's output include reduce axis
  right_product = right_product * reduce_product;
  // if after shape product < block_size, block_dim = 1
  if (right_product < block_size) {
    tilingInfo.block_tiling_axis = first_a_axis_index;
    tilingInfo.block_tiling_factor = input_shape[first_a_axis_index];
    return true;
  }
  std::size_t a_axis_count = 0;
  for (std::size_t i = 0; i < input_shape.size(); i++) {
    // block split A
    if (IsInVector(reduce_axis, i)) {
      right_product = right_product / input_shape[i];
      continue;
    }
    a_axis_count++;
    if (right_product / input_shape[i] <= block_size && left_product * input_shape[i] < compileInfo.core_num) {
      int64_t cur_block_factor =
          (block_size + right_product / input_shape[i] - 1) / (right_product / input_shape[i]);
      if (cur_block_factor <= input_shape[i]) {
        for (; cur_block_factor <= input_shape[i]; cur_block_factor++) {
          int64_t tail_block_factor =
              input_shape[i] % cur_block_factor == 0 ? cur_block_factor : input_shape[i] % cur_block_factor;
          int64_t tail_block_tilling_inner_ddr_count = tail_block_factor * (right_product / input_shape[i]);
          if (tail_block_tilling_inner_ddr_count >= block_size) {
            tilingInfo.block_tiling_axis = i;
            tilingInfo.block_tiling_factor = cur_block_factor;
            return true;
          }
        }
      } else {
        tilingInfo.block_tiling_axis = i;
        tilingInfo.block_tiling_factor = input_shape[i];
        return true;
      }
    } else if (right_product / input_shape[i] <= block_size || left_product * input_shape[i] >= compileInfo.core_num) {
      for (int32_t tilling_core_num = compileInfo.core_num; tilling_core_num <= left_product * input_shape[i];
           tilling_core_num += compileInfo.core_num) {
        if (left_product > tilling_core_num) {
          continue;
        }
        int64_t cur_block_dim = tilling_core_num / left_product;
        int64_t cur_block_factor = (input_shape[i] + cur_block_dim - 1) / cur_block_dim;
        int64_t tail_block_factor =
            input_shape[i] % cur_block_factor == 0 ? cur_block_factor : input_shape[i] % cur_block_factor;
        int64_t tail_block_tilling_inner_ddr_count = tail_block_factor * (right_product / input_shape[i]);
        if (tail_block_tilling_inner_ddr_count < block_size) {
          for (cur_block_dim = cur_block_dim - 1; cur_block_dim >= 1; cur_block_dim--) {
            cur_block_factor = (input_shape[i] + cur_block_dim - 1) / cur_block_dim;
            tail_block_factor =
                input_shape[i] % cur_block_factor == 0 ? cur_block_factor : input_shape[i] % cur_block_factor;
            tail_block_tilling_inner_ddr_count = tail_block_factor * (right_product / input_shape[i]);
            if (tail_block_tilling_inner_ddr_count >= block_size) {
              tilingInfo.block_tiling_axis = i;
              tilingInfo.block_tiling_factor = cur_block_factor;
              return true;
            }
          }
        } else {
          tilingInfo.block_tiling_axis = i;
          tilingInfo.block_tiling_factor = cur_block_factor;
          return true;
        }
      }
    }
    // before reduce shape product < core_num, which means split last a
    if (a_axis_count == input_shape.size() - reduce_axis.size()) {
      tilingInfo.block_tiling_axis = i;
      tilingInfo.block_tiling_factor = 1;
      return true;
    }
    left_product = left_product * input_shape[i];
    right_product = right_product / input_shape[i];
  }
  return false;
}

int32_t Norm::GetBlockDim(int32_t tiling_axis, int64_t tiling_factor)
{
  int32_t block_dim = 1;
  for (int32_t i = 0; i <= tiling_axis; i++) {
    if (IsInVector(reduce_axis, i)) {
      continue;
    }
    if (i == tiling_axis) {
      if (tiling_factor != 0) {
        block_dim = (int32_t)((input_shape[i] + tiling_factor - 1) / tiling_factor) * block_dim;
      }
    } else {
      block_dim = (int32_t)input_shape[i] * block_dim;
    }
  }
  return block_dim;
}

bool Norm::ProcessReorderAxis()
{
  /* InputShape: a0,r0,a1,r1,a2,r2,r3,a3
   *                    |---> block_tiling_axis
   *                    |---> core = a0*a1
   *                    |---> fused_block_tiling_axis= a0
   * ReorderShape: |a0,a1,a2|r0,r1,r2,r3|a3
   *                                   |---> last_r_axis_index
   * */
  int32_t block_tiling_axis = tilingInfo.block_tiling_axis;
  reorderInfo.reorder_input_shape.resize(input_shape.size());
  reorderInfo.reorderPos_oriPos.resize(input_shape.size());

  int num_r = reduce_axis.size();
  int num_a = input_shape.size() - num_r;
  for (const auto& item : reduce_axis) {
    reduce_flag[item] = 1;
  }
  // position of first R index in reorder shape
  int pos_r = num_a - ((int)input_shape.size() - (last_r_axis_index + 1));
  int pos_a = 0;

  // [0: last_r_axis_index]
  for (int32_t i = 0; i <= last_r_axis_index; i++) {
    if (reduce_flag[i] == 1) {
      reorderInfo.reorder_input_shape[pos_r] = input_shape[i];
      reorderInfo.reorderPos_oriPos[pos_r] = i;
      pos_r++;
    } else {
      if (i < block_tiling_axis) {
        reorderInfo.fused_block_tiling_axis.emplace_back(pos_a);
      }
      reorderInfo.reorder_input_shape[pos_a] = input_shape[i];
      reorderInfo.reorderPos_oriPos[pos_a] = i;
      pos_a++;
    }
  }

  // order last normal axis, maybe several axis
  for (size_t i = last_r_axis_index + 1; i < input_shape.size(); i++) {
    if ((int32_t)i < block_tiling_axis) {
      reorderInfo.fused_block_tiling_axis.emplace_back(pos_r);
    }
    reorderInfo.reorder_input_shape[pos_r] = input_shape[i];
    reorderInfo.reorderPos_oriPos[pos_r] = i;
    pos_r++;
  }

  return true;
}

int64_t Norm::CalcReorderShapeProduct(int32_t axis_index, int32_t block_tiling_axis_in_reorder)
{
  int64_t result = 1;

  for (uint32_t i = axis_index + 1; i < reorderInfo.reorder_input_shape.size(); i++) {
    if (IsInVector(reorderInfo.fused_block_tiling_axis, i)) {
      continue;
    }
    if (i == (uint32_t)block_tiling_axis_in_reorder) {
      result = result * tilingInfo.block_tiling_factor;
    } else {
      result = result * reorderInfo.reorder_input_shape[i];
    }
  }
  return result;
}

int64_t Norm::CalcReorderShapeProductAlign(int32_t axis_index, int32_t block_tiling_axis_in_reorder)
{
  int64_t result = 1;

  for (uint32_t i = axis_index + 1; i < reorderInfo.reorder_input_shape.size(); i++) {
    if (IsInVector(reorderInfo.fused_block_tiling_axis, i)) {
      continue;
    }

    if (i == (uint32_t)block_tiling_axis_in_reorder) {
      if (i == reorderInfo.reorder_input_shape.size() - 1) {
        result = result * ((tilingInfo.block_tiling_factor + block_size - 1) / block_size * block_size);
      } else {
        result = result * tilingInfo.block_tiling_factor;
      }
    } else {
      if (i == reorderInfo.reorder_input_shape.size() - 1) {
        result = result * ((reorderInfo.reorder_input_shape[i] + block_size - 1) / block_size * block_size);
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

bool Norm::PartialReorderUbTiling()
{
  if (!is_partial_reorder) {
    return false;
  }
  for (std::size_t i = 0; i < input_shape.size(); i++) {
    if (!IsInVector(reduce_axis, i)) {
      continue;
    }
    int64_t right_product_align = (i == input_shape.size() - 1 ?
                                   1:
                                   std::accumulate(input_align_shape.begin() + i + 1, input_align_shape.end(), 1,
                                                   std::multiplies<int64_t>()));
    if (right_product_align <= ub_size) {
      int64_t cur_ub_factor = ub_size / right_product_align;
      tilingInfo.ub_tiling_axis = (int32_t)i;
      tilingInfo.ub_tiling_factor = cur_ub_factor > input_shape[i] ? input_shape[i] : cur_ub_factor;
      return true;
    }
  }
  is_partial_reorder = false;
  return false;
}

bool Norm::GetUbTilingInfo()
{
  if (PartialReorderUbTiling()) {
    return true;
  }
  int32_t block_tiling_axis_in_reorder = -1;
  for (uint32_t i = 0; i < reorderInfo.reorderPos_oriPos.size(); i++) {
    if (reorderInfo.reorderPos_oriPos[i] == tilingInfo.block_tiling_axis) {
      block_tiling_axis_in_reorder = i;
      break;
    }
  }
  int32_t first_r_axis_in_reorder = last_r_axis_index - reduce_axis.size() + 1;
  int32_t last_r_axis_in_reorder = last_r_axis_index;
  int64_t right_product = 1;
  int64_t right_product_align = 1;
  // normal, ub split A
  if (!is_need_workspace) {
    for (int32_t i = 0; i < (int32_t)reorderInfo.reorder_input_shape.size(); i++) {
      if (IsInVector(reorderInfo.fused_block_tiling_axis, i)) {
        continue;
      }
      if (first_r_axis_in_reorder <= i && i <= last_r_axis_in_reorder) {
        continue;
      }
      right_product = CalcReorderShapeProduct(i, block_tiling_axis_in_reorder);
      right_product_align = CalcReorderShapeProductAlign(i, block_tiling_axis_in_reorder);
      int64_t current_dim = (i == block_tiling_axis_in_reorder) ?
                            tilingInfo.block_tiling_factor:
                            reorderInfo.reorder_input_shape[i];
      int64_t current_dim_tail = (i == block_tiling_axis_in_reorder) ?
                                 (reorderInfo.reorder_input_shape[i] % tilingInfo.block_tiling_factor):
                                 reorderInfo.reorder_input_shape[i];
      if (right_product_align <= ub_size) {
        int64_t cur_ub_factor = ub_size / right_product_align;
        for (; cur_ub_factor >= 1; cur_ub_factor--) {
          int64_t tail_ub_factor =
              current_dim % cur_ub_factor == 0 ? cur_ub_factor : current_dim % cur_ub_factor;
          int64_t tail_ub_tilling_inner_ddr_count = tail_ub_factor * right_product;
          int64_t tail_tail_ub_factor =
              current_dim_tail % tail_ub_factor == 0 ? cur_ub_factor : current_dim_tail % cur_ub_factor;
          int64_t tail_tail_ub_tilling_inner_ddr_count = tail_tail_ub_factor * right_product;
          if (tail_ub_tilling_inner_ddr_count >= block_size && tail_tail_ub_tilling_inner_ddr_count >= block_size) {
            tilingInfo.ub_tiling_axis = reorderInfo.reorderPos_oriPos[i];
            int64_t ub_loop = cur_ub_factor > current_dim ? 1 : (current_dim + cur_ub_factor - 1) / cur_ub_factor;
            tilingInfo.ub_tiling_factor = (current_dim + ub_loop - 1) / ub_loop;
            if (tilingInfo.ub_tiling_factor > ub_size / right_product_align) {
              tilingInfo.ub_tiling_factor = cur_ub_factor > current_dim ? current_dim : cur_ub_factor;
            }
            // storage align last A when is nlast reduce, but align_factor * right_product is larger than max_ub
            if (!(!is_last_axis_reduce && i == (int32_t)reorderInfo.reorder_input_shape.size() - 1 &&
                  (tilingInfo.ub_tiling_factor + block_size - 1) / block_size * block_size * right_product_align >
                  ub_size)) {
              return true;
            }
          }
        }
      }
    }
    tilingInfo.ub_tiling_axis = first_a_axis_index;
    tilingInfo.ub_tiling_factor = input_shape[first_a_axis_index];
    return true;
  } else {
    // workspace, ub split R
    for (int32_t i = 0; i < (int32_t)reorderInfo.reorder_input_shape.size(); i++) {
      if (IsInVector(reorderInfo.fused_block_tiling_axis, i)) {
        continue;
      }
      if (!(first_r_axis_in_reorder <= i && i <= last_r_axis_in_reorder)) {
        continue;
      }
      right_product = CalcReorderShapeProduct(i, block_tiling_axis_in_reorder);
      right_product_align = CalcReorderShapeProductAlign(i, block_tiling_axis_in_reorder);
      int64_t current_dim = reorderInfo.reorder_input_shape[i];
      if (right_product_align <= ub_size) {
        int64_t cur_ub_factor = ub_size / right_product_align;
        for (; cur_ub_factor >= 1; cur_ub_factor--) {
          int64_t tail_ub_factor =
              current_dim % cur_ub_factor == 0 ? cur_ub_factor : current_dim % cur_ub_factor;
          int64_t tail_ub_tilling_inner_ddr_count = tail_ub_factor * right_product;
          if (tail_ub_tilling_inner_ddr_count >= block_size) {
            tilingInfo.ub_tiling_axis = reorderInfo.reorderPos_oriPos[i];
            int64_t ub_loop = cur_ub_factor > current_dim ? 1 : (current_dim + cur_ub_factor - 1) / cur_ub_factor;
            tilingInfo.ub_tiling_factor = (current_dim + ub_loop - 1) / ub_loop;
            if (tilingInfo.ub_tiling_factor > ub_size / right_product_align) {
              tilingInfo.ub_tiling_factor = cur_ub_factor > current_dim ? current_dim : cur_ub_factor;
            }
            // storage align last R when is last reduce, but align_factor * right_product is larger than max_ub
            if (!(is_last_axis_reduce && i == (int32_t)reorderInfo.reorder_input_shape.size() - 1 &&
                  (tilingInfo.ub_tiling_factor + block_size - 1) / block_size * block_size * right_product_align >
                  ub_size)) {
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}

// block split last common axis, block factor can refine to align
bool Norm::NeedRefineBlockTiling()
{
  if (is_last_axis_reduce || tilingInfo.block_tiling_axis != (int32_t)input_shape.size() - 1) {
    return false;
  }
  int64_t refined_block_tiling_factor = (tilingInfo.block_tiling_factor + block_size - 1) / block_size * block_size;
  if (is_need_workspace) {
    if (refined_block_tiling_factor > ub_size) {
      return false;
    }
  }
  int64_t tail_dim = input_shape.back() % refined_block_tiling_factor == 0 ?
                     refined_block_tiling_factor:
                     input_shape.back() % refined_block_tiling_factor;
  if (tail_dim < block_size) {
    return false;
  }

  return true;
}

bool Norm::ProcessTiling()
{
  bool ret = true;
  if (is_split_block) {
    ret = ret && GetBlockTilingInfo();
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetBlockTilingInfo error");
      return false;
    }
    if (NeedRefineBlockTiling()) {
      tilingInfo.block_tiling_factor = (tilingInfo.block_tiling_factor + block_size - 1) / block_size * block_size;
    }
    tilingInfo.block_dim = GetBlockDim(tilingInfo.block_tiling_axis, tilingInfo.block_tiling_factor);
  } else {
    tilingInfo.block_dim = 1;
  }
  ret = ret && ProcessReorderAxis();
  ret = ret && GetUbTilingInfo();
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetUbTilingInfo error");
    return false;
  }
  return ret;
}

bool Norm::GetVarValue()
{
  std::size_t count_var = 0;
  // obtain var value
  if (!compileInfo.is_const) {
    try {
      const auto& var_pattern = op_info.at("_norm_vars").at(std::to_string(tiling_key)).get<std::vector<int32_t>>();
      for (const auto& var : var_pattern) {
        if (var >= 300) {
          var_value[count_var] = (int32_t)tilingInfo.ub_tiling_factor;
          count_var++;
        } else if (var >= 200) {
          var_value[count_var] = (int32_t)tilingInfo.block_tiling_factor;
          count_var++;
        } else {
          var_value[count_var] = (int32_t)input_shape[var % 100];
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

bool Norm::DoTiling()
{
  /* Situations of DoTiling include:
     1. input(known):
        status of compile: do others except FusedReduceAxis
        status of runtime: do WriteTilingData
     2. input(unknown):
        do all process
  */
  bool ret = Init();

  if (compileInfo.is_const) {
    // input(known)
    // invoking the tiling interface during runtime
    // is_const_post: "true"->runtime, "false"->compile
    try {
      compileInfo.is_const_post = op_info.at("_const_shape_post").get<bool>();
      if (compileInfo.is_const_post) {
        return ret;
      } else {
        reduce_axis = reduce_axis_ori;
        input_shape = input_shape_ori;
      }
    } catch (const std::exception &e) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info error. Error message: %s", e.what());
      return false;
    }
  } else if (!compileInfo.is_fuse_axis) {
    reduce_axis = reduce_axis_ori;
    input_shape = input_shape_ori;
  } else {
    // input(unknown)
    ret = ret && FusedReduceAxis();
  }

  ret = ret && GetCompileInfo();
  // calculate some variables
  block_size = compileInfo.min_block_size;

  for (std::size_t i = 0; i < input_shape.size(); i++) {
    if (i == input_shape.size() - 1) {
      input_align_shape[i] = (input_shape[i] + block_size - 1) / block_size * block_size;
      continue;
    }
    input_align_shape[i] = input_shape[i];
  }
  input_align_shape.resize(input_shape.size());

  shape_after_reduce_product = CalcAfterReduceShapeProduct(input_shape, reduce_axis);
  reduce_product = CalcReduceShapeProduct(input_shape, reduce_axis);

  for (int32_t i = 0; i < (int32_t)input_shape.size(); i++) {
    if (!IsInVector(reduce_axis, i)) {
      first_a_axis_index = i;
      break;
    }
  }
  last_r_axis_index = (int32_t)*max_element(reduce_axis.begin(), reduce_axis.end());
  is_last_axis_reduce = last_r_axis_index == (int32_t)input_shape.size() - 1;
  is_split_block = !(reduce_axis.size() == input_shape.size());
  is_partial_reorder = reduce_axis.size() > 1 && input_shape.back() < block_size && is_split_block;
  pattern = CalcPattern(input_shape, reduce_axis);
  is_need_workspace = IsNeedWorkspace();
  is_align_and_remove_pad = !is_need_workspace && input_shape.size() == 2 && is_last_axis_reduce &&
      input_shape.back() % block_size != 0 &&
      input_shape[0] != 1 && input_shape[1] != 1 &&
      compileInfo.pad_max_ub_count / compileInfo.pad_max_entire_size >= input_align_shape.back();
  // determine ub size
  if (is_need_workspace) {
    ub_size = compileInfo.workspace_max_ub_count;
  } else if (is_align_and_remove_pad) {
    ub_size = compileInfo.pad_max_ub_count;
    sch_type = 4;
  } else {
    ub_size = compileInfo.max_ub_count;
  }
  // calculate tiling info
  ret = ret && ProcessTiling();
  // calculate workspace size
  ret = ret && CalcWorkspace();
  // calculate tiling key
  ret = ret && CalcTilingKey();
  // get var value
  ret = ret && GetVarValue();

  return ret;
}

bool Norm::CalcTilingKey()
{
  std::vector<int> pos;
  std::vector<int> coefficient;
  if (is_split_block) {
    pos = {db, sch_type, tilingInfo.block_tiling_axis, tilingInfo.ub_tiling_axis, pattern};
    coefficient = {1000000000, 10000000, 1000000, 100000, 100};
  } else {
    pos = {db, sch_type, tilingInfo.ub_tiling_axis, pattern};
    coefficient = {1000000000, 10000000, 1000000, 1000};
  }
  int32_t key = 0;
  for (std::size_t i = 0; i < coefficient.size(); i++) {
    key += pos[i] * coefficient[i];
  }
  tiling_key = key;

  return true;
}

bool Norm::CalcWorkspace()
{
  const auto& workspace_type = op_info.at("_workspace_info").at("_workspace_type").get<std::vector<int32_t>>();
  const auto& workspace_bytes = op_info.at("_workspace_info").at("_workspace_bytes").get<std::vector<int32_t>>();
  const auto& workspace_diff_count = op_info.at("_workspace_info").at("_workspace_diff_count").get<int32_t>();
  // CHECK VALUE
  V_CHECK_EQ(workspace_type.size(), workspace_bytes.size(),
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "size of workspace_type and workspace_bytes should be equal"),
             return false);
  V_OP_TILING_CHECK((workspace_diff_count >= 0 && (uint32_t)workspace_diff_count <= workspace_type.size()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "workspace_diff_count is %d that is illegal",
                                                    workspace_diff_count),
                    return false);

  std::size_t workspace_count = workspace_type.size();
  if (workspace_count != 0) {
    if (is_need_workspace) {
      int64_t shape_after_reduce_align_product = CalcAfterReduceShapeProduct(input_align_shape, reduce_axis);
      int64_t shape_before_reduce_align_product = std::accumulate(input_align_shape.begin(),
                                                                  input_align_shape.end(),
                                                                  1,
                                                                  std::multiplies<int64_t>());
      for (std::size_t i = 0; i < workspace_count; i++) {
        // workspace sch may need fake workspace
        if (!is_partial_reorder && i >= workspace_count - (int32_t)workspace_diff_count) {
          workspace[i] = 32;
        } else {
          if (workspace_type[i] == 1) {
            workspace[i] = shape_before_reduce_align_product * workspace_bytes[i];
          } else {
            workspace[i] = shape_after_reduce_align_product * workspace_bytes[i];
          }
        }
      }
    } else if (!compileInfo.is_const) {
      for (std::size_t i = 0; i < workspace_count; i++) {
        workspace[i] = 32;
      }
    }
  }
  workspace.resize(workspace_count);
  return true;
}

bool Norm::ConstInputProcPost()
{
  // runtime
  try {
    int32_t const_tiling_key = op_info.at("_const_tiling_key").get<std::int32_t>();
    workspace = op_info.at("_const_workspace_size").get<std::vector<std::int64_t>>();
    run_info.SetBlockDim(op_info.at("_block_dims").get<std::int32_t>());
    run_info.SetTilingKey(const_tiling_key);
    for (const auto& item : workspace) {
      run_info.AddWorkspace(item);
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Func: ConstInputProcPost get error message. Error message: %s",
                                    e.what());
    return false;
  }

  return true;
}

bool Norm::WriteTilingData()
{
  if (compileInfo.is_const_post) {
    // runtime
    return ConstInputProcPost();
  }

  if (compileInfo.is_const) {
    // compile
    if (is_split_block) {
      run_info.AddTilingData((int32_t)tilingInfo.block_tiling_axis);
      run_info.AddTilingData((int32_t)tilingInfo.block_tiling_factor);
    }
    run_info.AddTilingData((int32_t)tilingInfo.ub_tiling_axis);
    run_info.AddTilingData((int32_t)tilingInfo.ub_tiling_factor);
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

bool NormTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                utils::OpRunInfo& run_info)
{
  OP_LOGD(op_type.c_str(), "norm tiling running");
  Norm norm(op_type, op_paras, op_info, run_info);
  bool ret = norm.GetInput();
  ret = ret && norm.DoTiling();
  ret = ret && norm.WriteTilingData();
  return ret;
}

}  // namespace optiling
