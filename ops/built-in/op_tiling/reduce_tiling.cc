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
 * \file reduce_tiling.cpp
 * \brief tiling function of op
 */

#include <algorithm>
#include "reduce_tiling.h"

namespace optiling {

std::vector<int64_t> Reduce::GetInputShape() {
  return input_shape;
}

std::vector<int32_t> Reduce::GetReduceAxis() {
  return reduce_axis;
}

bool Reduce::IsInVector(std::vector<int32_t>& input, int32_t value) {
  for (uint32_t i = 0; i < input.size(); i++) {
    if (input[i] == value) {
      return true;
    }
  }
  return false;
}

int32_t Reduce::CalcPattern(std::vector<int64_t>& input, std::vector<int32_t>& axis) {
  int32_t pattern = 0;
  for (size_t i = 0; i < input.size(); i++) {
    if (IsInVector(axis, i)) {
      pattern += 2 << (input.size() - i - 1);
    } else {
      pattern += ((int)input.size() - 2 - (int)i) >= 0 ? 2 << (input.size() - 2 - i) : 1;
    }
  }
  return pattern;
}

int32_t Reduce::CalcConstPattern(std::vector<int32_t>& reduce_axis) {
  // generate dict key according to reduce_axis
  // Init() make reduce axis sorted
  if (reduce_axis.size() == 0) {
    return -1;
  }

  int32_t dict_key = 0;
  for (auto& i : reduce_axis) {
    // dict_key: 1234 -> reduce [0,1,2,3]
    dict_key = 10 * dict_key + i + 1;
  }

  return dict_key;
}

int64_t Reduce::GetReorderInputShapeMul(int32_t axis_index, int32_t block_tiling_axis_in_reorder) {
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
      result = result * reorderInfo.reorder_input_shape[i];
    }
  }
  return result;
}

int64_t Reduce::GetAlignShapeMul(int32_t axis_index) {
  int64_t result = 1;
  for (uint32_t i = axis_index + 1; i < output_shape.size(); i++) {
    if (output_shape[i] == 0) {
      continue;
    }
    if (i == output_shape.size() - 1 && !is_last_axis_reduce) {
      result = result * ((output_shape[i] + block_size - 1) / block_size * block_size);
    } else {
      result = result * output_shape[i];
    }
  }
  return result;
}

int64_t Reduce::GetShapeMul(std::vector<int64_t>& shape, int32_t axis_index) {
  int64_t result = 1;
  for (uint32_t i = axis_index + 1; i < shape.size(); i++) {
    if (shape[i] != 0) {
      result = result * shape[i];
    }
  }
  return result;
}

int32_t Reduce::GetBlockDim(std::vector<int64_t>& out, int32_t tiling_axis, int64_t tiling_factor) {
  int32_t block_dim = 1;
  for (int32_t i = 0; i <= tiling_axis; i++) {
    if (out[i] != 0) {
      if (i == tiling_axis) {
        block_dim = (int32_t)((out[i] + tiling_factor - 1) / tiling_factor) * block_dim;
      } else {
        block_dim = (int32_t)out[i] * block_dim;
      }
    }
  }
  return block_dim;
}

int32_t Reduce::GetRealBlockTilingAxis(std::vector<int64_t>& shape, int32_t idx) {
  int32_t zero_cnt = 0;
  for (int32_t i = 0; i < idx; i++) {
    if (shape[i] == 0) {
      zero_cnt += 1;
    }
  }
  return idx - zero_cnt;
}

int32_t Reduce::CalcTilingKey() {
  using namespace std;
  int db = 0;
  int shape_type = 0;
  vector<int> pos = {db, shape_type, tilingInfo.block_tiling_axis, tilingInfo.ub_tiling_axis, pattern};
  vector<int> coefficient = {1000000000, 10000000, 1000000, 100000, 100};
  int32_t key = 0;
  for (size_t i = 0; i < coefficient.size(); i++) {
    key += pos[i] * coefficient[i];
  }
  key = compileInfo.atomic ? key * 1 : -1 * key;
  return key;
}

void Reduce::EliminateOne() {
  for (auto item : reduce_axis_ori) {
    reduce_flag[item] = 1;
  }

  size_t skip_count = 0;
  size_t pos_a = 0;
  size_t pos_r = 0;
  for (size_t i = 0; i < input_shape_ori.size(); i++) {
    if (input_shape_ori[i] == 1) {
      skip_count++;
      continue;
    }

    normalize_shape[pos_a] = input_shape_ori[i];
    pos_a++;
    if (reduce_flag[i] == 1) {
      normalize_axis[pos_r] = i - skip_count;
      pos_r++;
    }
  }

  if (pos_a < input_shape_ori.size()) {
    normalize_shape.resize(pos_a);
    input_shape_ori = normalize_shape;
  }
  // reset flag
  for (auto item : reduce_axis_ori) {
    reduce_flag[item] = 0;
  }
  // sort axis
  normalize_axis.resize(pos_r);
  reduce_axis_ori = normalize_axis;
  if (input_shape_ori.empty()) {
    input_shape_ori.emplace_back(1);
  }
}

bool Reduce::ConstInputProcPost() {
  // runtime
  std::string pattern_str = std::to_string(pattern);
  try {
    run_info.block_dim = op_info.at("_block_dims").at(pattern_str).get<std::int32_t>();
    run_info.clear_atomic = op_info.at("_atomic_flags").at(pattern_str).get<bool>();
    run_info.tiling_key = pattern;
    int status = op_info.at("push_status");
    if (status == 0) {
      ByteBufferPut(run_info.tiling_data, pattern);
    }
  } catch (const std::exception &e) {
    OP_LOGE(op_type.c_str(), "Func: ConstInputProcPost get error message. Error message: %s", e.what());
    return false;
  }

  return true;
}

bool Reduce::FusedReduceAxis() {
  /* model: reduce_axis_unknown (eg: reduce_sum)
   * if input_shape_ori is [-1], model is R
   * if input_shape_ori is [-1, -1], model is AR, RA
   * if input_shape_ori is [-1, -1, -1], model is AR, ARA, RAR
   * if input_shape_ori is [-1, -1, -1, -1], model is AR, ARA, ARAR, RARA
   * Special Regulation:
   * if after fused, model is RA while len(input_shape_ori) is 3, model will be ARA by padding "1".
   * if after fused, model is R while len(input_shape_ori) isn't 1, model will be AR by padding "1".
   * if after fused, model is A, model will be RA by padding "1".
   * */
  vector<int32_t> pos(input_shape_ori.size());
  for (auto item : reduce_axis_ori) {
    pos[item] = 1;
  }

  // Fused serial axises which in same type.
  size_t first = 0;
  size_t second = 0;
  int64_t value_input = 0;
  int32_t value_axis = 0;
  size_t length = input_shape_ori.size();
  bool cond_0 = false;
  bool cond_1 = false;

  size_t capacity_shape = 0;
  size_t capacity_axis = 0;

  // Deal Model
  if (reduce_axis_ori.size() == 0) {
    // model is A, A -> RA (only one RA)
    input_shape[0] = 1;
    reduce_axis[0] = 0;
    capacity_shape++;
    capacity_axis++;
  } else if (reduce_axis_ori[0] == 0) {
    // model is Rx, Rx -> ARx
    input_shape[0] = 1;
    capacity_shape++;
  }

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
  return true;
}

bool Reduce::GetCompileInfo() {
  std::vector<int32_t> common_info;
  std::vector<int32_t> pattern_info;
  std::vector<int32_t> ub_info;

  try {
    common_info = op_info.at("_common_info").get<std::vector<int32_t>>();
    pattern_info = op_info.at("_pattern_info").get<std::vector<int32_t>>();
    ub_info = op_info.at("_ub_info").get<std::vector<int32_t>>();

    const int32_t common_info_size = 5;
    V_CHECK_EQ(common_info.size(), common_info_size,
               OP_LOGE(op_type.c_str(), "size of common_info should be 5"),
               return false);
    compileInfo.core_num = common_info[0];
    compileInfo.is_keep_dims = (bool)common_info[1];
    compileInfo.min_block_size = common_info[2];
    compileInfo.atomic = (bool)common_info[3];
    compileInfo.coef = common_info[4];
    // CHECK VALUE
    V_OP_TILING_CHECK(!(compileInfo.coef <= 0),
                      OP_LOGE(op_type.c_str(), "coef is %d that is illegal", compileInfo.coef),
                      return false);
    V_OP_TILING_CHECK(!(compileInfo.min_block_size <= 0),
                      OP_LOGE(op_type.c_str(), "min_block_size is %d that is illegal", compileInfo.min_block_size),
                      return false);
    V_OP_TILING_CHECK(!(compileInfo.core_num <= 0),
                      OP_LOGE(op_type.c_str(), "core_num is %d that is illegal", compileInfo.core_num),
                      return false);

    uint idx = 0;
    for (auto item : pattern_info) {
      if (item == pattern) {
        break;
      }
      idx += 1;
    }
    if (idx >= pattern_info.size()) {
      OP_LOGE(op_type.c_str(), "pattern is %d that not in pattern_info", pattern);
      return false;
    }

    V_CHECK_EQ(pattern_info.size(), ub_info.size(),
               OP_LOGE(op_type.c_str(), "pattern_info's size should be as same as ub_info"),
               return false);
    compileInfo.max_ub_count = ub_info[idx];
    V_OP_TILING_CHECK(!(compileInfo.max_ub_count <= 0),
                      OP_LOGE(op_type.c_str(), "max_ub_count is %d that is illegal", compileInfo.max_ub_count),
                      return false);
    is_last_axis_reduce = IsInVector(reduce_axis, input_shape.size() - 1);

  } catch (const std::exception &e) {
    OP_LOGE(op_type.c_str(), "Func: GetCompileInfo error. Error message: %s", e.what());
    return false;
  }

  return true;
}

bool Reduce::ChooseAtomic() {
  total_output_count = 1;
  total_reduce_count = 1;
  output_shape.resize(input_shape.size());

  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      if (compileInfo.is_keep_dims) {
        output_shape[i] = 1;
      } else {
        output_shape[i] = 0;
      }
      total_reduce_count *= input_shape[i];
    } else {
      output_shape[i] = input_shape[i];
      total_output_count *= input_shape[i];
    }
  }

  // block_size: would be used for storage align in whole graph(min dtype)
  // nodes after reduce(include reduce) have same space(ubSizeA)
  // nodes before reduce have same space(ubSizeB)
  // ubSizeB = ubSizeA * coef (max dtype)
  block_size = compileInfo.min_block_size;
  ubSizeB = compileInfo.max_ub_count;
  ubSizeA = ubSizeB / compileInfo.coef;

  // 1: Check if atomic is enabled
  // 2: Check if output is large enough (> SMALL_SHAPE_THRESHOLD)
  // 3: Check normal atomic rules
  // 4: (Priority) Check if it is outermost_reduce and is larger than or equal to core_num
  // Layer 0 (Required)
  bool atomic_available = compileInfo.atomic;
  // Layer 1
  compileInfo.atomic = total_output_count <= ubSizeB &&
                       total_output_count * total_reduce_count > SMALL_SHAPE_THRESHOLD &&
                       total_output_count < (int64_t)compileInfo.core_num * block_size / 2 &&
                        total_reduce_count > (int64_t)compileInfo.core_num / 2;
  // Layer 2
  // Currently unavailable

  // Final
  compileInfo.atomic = atomic_available && compileInfo.atomic;
  return true;
}

bool Reduce::GetUbTilingInfo() {
  // rewrite ub_tiling_factor, ub_tiling_axis
  int32_t block_tiling_axis_in_reorder = -1;
  for (uint32_t i = 0; i < reorderInfo.reorderPos_oriPos.size(); i++) {
    if (reorderInfo.reorderPos_oriPos[i] == tilingInfo.block_tiling_axis) {
      block_tiling_axis_in_reorder = i;
      break;
    }
  }

  int64_t load_mul = 1;
  for (int32_t i = 0; i < (int32_t)reorderInfo.reorder_input_shape.size(); i++) {
    if (IsInVector(reorderInfo.fused_block_tiling_axis, i)) {
      continue;
    }

    load_mul = GetReorderInputShapeMul(i, block_tiling_axis_in_reorder);
    if (load_mul <= ubSizeB) {
      tilingInfo.ub_tiling_axis = reorderInfo.reorderPos_oriPos[i];
      tilingInfo.ub_tiling_factor = (ubSizeB / load_mul);

      int64_t max_ub_tiling_factor = input_shape[tilingInfo.ub_tiling_axis];
      if (i == block_tiling_axis_in_reorder) {
        max_ub_tiling_factor = tilingInfo.block_tiling_factor;
      }
      if (tilingInfo.ub_tiling_factor > max_ub_tiling_factor) {
        tilingInfo.ub_tiling_factor = max_ub_tiling_factor;
      }
      return true;
    }
  }

  return false;
}

void Reduce::ProcessReorderAxis(int32_t fused_type) {
  /* InputShape: a0,r0,a1,r1,a2,r2,r3,a3
   *                    |---> block_tiling_axis(NormalReduce)
   *                    |---> core = a0*a1
   *                                   |--->fused_block_tiling_axis
   * ReorderShape: |a0,a1,a2|r0,r1,r2,r3|a3
   *                                   |---> last_reduce_axis_idx
   * */
  int32_t block_tiling_axis = tilingInfo.block_tiling_axis;
  int32_t last_reduce_axis_idx = reduce_axis.back();
  reorderInfo.reorder_input_shape.resize(input_shape.size());
  reorderInfo.reorderPos_oriPos.resize(input_shape.size());

  int num_r = reduce_axis.size();
  int num_a = input_shape.size() - num_r;
  for (auto item : reduce_axis) {
    reduce_flag[item] = 1;
  }

  int pos_r = num_a - ((int)input_shape.size() - (last_reduce_axis_idx + 1));
  int pos_a = 0;

  // [0: last_reduce_axis_idx]
  for (int32_t i = 0; i <= last_reduce_axis_idx; i++) {
    if (reduce_flag[i] == 1) {
      if (fused_type == FUSED_REDUCE_AXIS && i < block_tiling_axis) {
        reorderInfo.fused_block_tiling_axis.emplace_back(pos_r);
      }
      reorderInfo.reorder_input_shape[pos_r] = input_shape[i];
      reorderInfo.reorderPos_oriPos[pos_r] = i;
      pos_r++;
    } else {
      if (fused_type == FUSED_NON_REDUCE_AXIS && i < block_tiling_axis) {
        reorderInfo.fused_block_tiling_axis.emplace_back(pos_a);
      }
      reorderInfo.reorder_input_shape[pos_a] = input_shape[i];
      reorderInfo.reorderPos_oriPos[pos_a] = i;
      pos_a++;
    }
  }

  // order last non axis, maybe several axis
  for (size_t i = last_reduce_axis_idx + 1; i < input_shape.size(); i++) {
    if (fused_type == FUSED_NON_REDUCE_AXIS && (int32_t)i < block_tiling_axis) {
      reorderInfo.fused_block_tiling_axis.emplace_back(pos_r);
    }
    reorderInfo.reorder_input_shape[pos_r] = input_shape[i];
    reorderInfo.reorderPos_oriPos[pos_r] = i;
    pos_r++;
  }

  return;
}

bool Reduce::GetAtomicBlockDim() {
  // reload block_dim
  int32_t block_dim = 1;
  for (int32_t i = 0; i <= tilingInfo.block_tiling_axis; i++) {
    if (IsInVector(reduce_axis, i)) {
      if (i == tilingInfo.block_tiling_axis) {
        block_dim = (int32_t)((input_shape[i] + tilingInfo.block_tiling_factor - 1) / tilingInfo.block_tiling_factor) *
                    block_dim;
        break;
      } else {
        block_dim = (int32_t)input_shape[i] * block_dim;
      }
    }
  }
  // > 65535 -> false
  tilingInfo.block_dim = block_dim;
  return true;
}

bool Reduce::GetAtomicBlockTilingInfo() {
  // rewrite block_tiling_axis, block_tiling_factor.
  bool is_find_block_tiling = false;
  int64_t left_mul = 1;
  int32_t core_num = compileInfo.core_num;
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      is_find_block_tiling = true;
      tilingInfo.block_tiling_axis = i;
      tilingInfo.block_tiling_factor = 1;

      if (left_mul * input_shape[i] >= core_num) {
        tilingInfo.block_tiling_axis = i;
        int64_t block_tiling_factor_outer = core_num / left_mul;
        tilingInfo.block_tiling_factor = (input_shape[i] + block_tiling_factor_outer - 1) / block_tiling_factor_outer;
        return is_find_block_tiling;
      }
      left_mul = left_mul * input_shape[i];
    }
  }

  return is_find_block_tiling;
}

void Reduce::GetNotMulCoreBlockTiling() {
  if (input_shape.size() <= 0) {
    return;
  }
  tilingInfo.block_tiling_axis = 0;
  tilingInfo.block_tiling_factor = input_shape[0];
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (!IsInVector(reduce_axis, i)) {
      tilingInfo.block_tiling_axis = i;
      tilingInfo.block_tiling_factor = input_shape[i];
      return;
    }
  }
  return;
}

bool Reduce::GetBlockTilingInfo() {
  int32_t left_block_dim = 1;
  int64_t max_ub_count = ubSizeA;
  int32_t core_num = compileInfo.core_num;

  for (uint32_t i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] == 0 || output_shape[i] == 1) {
      // block_split not in reduce_axis
      continue;
    }

    // right_inner_ub_count: prod(output[i+1:])
    int64_t right_inner_ub_count = GetAlignShapeMul(i);
    if (right_inner_ub_count > max_ub_count) {
      left_block_dim = (int32_t)left_block_dim * output_shape[i];
      continue;
    }

    // max_block_tiling_factor: UB can store m * inner_ub_count
    int64_t max_block_tilling_factor = max_ub_count / right_inner_ub_count;
    int64_t right_total_num = GetShapeMul(output_shape, i);

    // case0: prod(output[i+1:]) <= block_size && core < core_num
    // case1: prod(output[i+1:]) <= block_size && core > block_size
    // case2: prod(output[i+1:]) > block_size && core > block_size
    // case3: prod(output[i+1:]) > block_size && core < core_num
    if (right_total_num <= block_size && left_block_dim * output_shape[i] < core_num) {
      if (left_block_dim > 1) {
        int64_t cur_block_factor = (block_size + right_total_num - 1) / right_total_num;
        if (cur_block_factor <= output_shape[i]) {
          for (; cur_block_factor <= output_shape[i]; cur_block_factor++) {
            int64_t tail_block_factor =
                output_shape[i] % cur_block_factor == 0 ? cur_block_factor : output_shape[i] % cur_block_factor;
            int64_t tail_block_tilling_inner_ddr_count = tail_block_factor * right_total_num;
            if (tail_block_tilling_inner_ddr_count >= block_size) {
              tilingInfo.block_tiling_axis = i;
              tilingInfo.block_tiling_factor = cur_block_factor;
              return true;
            }
          }
        } else {
          tilingInfo.block_tiling_axis = i;
          tilingInfo.block_tiling_factor = output_shape[i];
          return true;
        }
      } else {
        int64_t cur_block_factor = (block_size + right_total_num - 1) / right_total_num;
        tilingInfo.block_tiling_axis = i;
        tilingInfo.block_tiling_factor = cur_block_factor;
        return true;
      }
    } else if (right_total_num <= block_size || left_block_dim * output_shape[i] >= core_num) {
      for (int32_t tilling_core_num = core_num; tilling_core_num <= left_block_dim * output_shape[i];
           tilling_core_num += core_num) {
        if (left_block_dim > tilling_core_num) {
          continue;
        }
        int64_t cur_block_dim = tilling_core_num / left_block_dim;
        int64_t cur_block_factor = (output_shape[i] + cur_block_dim - 1) / cur_block_dim;
        if (cur_block_factor > max_block_tilling_factor) {
          continue;
        }

        if (left_block_dim > 1) {
          int64_t tail_block_factor =
              output_shape[i] % cur_block_factor == 0 ? cur_block_factor : output_shape[i] % cur_block_factor;
          int64_t tail_block_tilling_inner_ddr_count = tail_block_factor * right_total_num;
          if (tail_block_tilling_inner_ddr_count < block_size) {
            for (cur_block_dim = cur_block_dim - 1; cur_block_dim >= 1; cur_block_dim--) {
              cur_block_factor = (output_shape[i] + cur_block_dim - 1) / cur_block_dim;
              tail_block_factor =
                  output_shape[i] % cur_block_factor == 0 ? cur_block_factor : output_shape[i] % cur_block_factor;
              tail_block_tilling_inner_ddr_count = tail_block_factor * right_total_num;
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

        } else {
          int64_t block_tilling_inner_ddr_count = cur_block_factor * right_total_num;
          if (block_tilling_inner_ddr_count < block_size) {
            cur_block_factor = (block_size + right_total_num - 1) / right_total_num;
          }

          tilingInfo.block_tiling_axis = i;
          tilingInfo.block_tiling_factor = cur_block_factor;
          return true;
        }
      }
    }

    left_block_dim = (int32_t)left_block_dim * output_shape[i];
  }

  return false;
}

bool Reduce::ProcessAtomicTiling() {
  // init
  tilingInfo.block_dim = 0;
  tilingInfo.block_tiling_axis = 0;
  tilingInfo.block_tiling_factor = 0;
  tilingInfo.ub_tiling_axis = 0;
  tilingInfo.ub_tiling_factor = 0;

  // rewrite TilingInfo(block)
  if (!GetAtomicBlockTilingInfo()) {
    return false;
  }
  if (!GetAtomicBlockDim()) {
    return false;
  }

  // rewrite ReorderInfo
  ProcessReorderAxis(FUSED_REDUCE_AXIS);

  // align
  if (block_size != 0 && reorderInfo.reorder_input_shape.size() > 0 &&
      reorderInfo.reorder_input_shape.back() % block_size != 0) {
    reorderInfo.reorder_input_shape.back() =
        (reorderInfo.reorder_input_shape.back() + block_size - 1) / block_size * block_size;
  }

  // rewrite TilingInfo(ub)
  return GetUbTilingInfo();
}

bool Reduce::ProcessNormalTiling() {
  // init
  tilingInfo.block_dim = 1;
  tilingInfo.block_tiling_axis = 0;
  tilingInfo.block_tiling_factor = input_shape[0];
  tilingInfo.ub_tiling_axis = 0;
  tilingInfo.ub_tiling_factor = input_shape[0];

  // rewrite TilingInfo(block)
  if (!GetBlockTilingInfo()) {
    if (total_output_count > ubSizeA) {
      OP_LOGE(op_type.c_str(), "GetBlockTilingInfo error");
      return false;
    }
    tilingInfo.block_dim = 1;
    GetNotMulCoreBlockTiling();
  } else {
    tilingInfo.block_dim = GetBlockDim(output_shape, tilingInfo.block_tiling_axis, tilingInfo.block_tiling_factor);
  }

  // rewrite ReorderInfo
  ProcessReorderAxis(FUSED_NON_REDUCE_AXIS);
  // align
  if (block_size != 0 && reorderInfo.reorder_input_shape.size() > 0 &&
      reorderInfo.reorder_input_shape.back() % block_size != 0) {
    reorderInfo.reorder_input_shape.back() =
        (reorderInfo.reorder_input_shape.back() + block_size - 1) / block_size * block_size;
  }

  // rewrite TilingInfo(ub)
  tilingInfo.ub_tiling_axis = tilingInfo.block_tiling_axis;
  tilingInfo.ub_tiling_factor = tilingInfo.block_tiling_factor;
  if (!GetUbTilingInfo()) {
    return false;
  }

  if (!compileInfo.is_keep_dims) {
    tilingInfo.block_tiling_axis = GetRealBlockTilingAxis(output_shape, tilingInfo.block_tiling_axis);
  }

  return true;
}

bool Reduce::SpecialUBTiling() {
  int32_t ub_factor = tilingInfo.ub_tiling_factor;
  int32_t blk_factor = tilingInfo.block_tiling_factor;
  int32_t ub_axis = tilingInfo.ub_tiling_axis;
  int32_t blk_axis = tilingInfo.block_tiling_axis;

  V_OP_TILING_CHECK(!(ub_factor <= 0 || blk_factor <= 0),
                    OP_LOGE(op_type.c_str(), "ub_factor and blk_factor must bigger than 0,"
                            "while ub_factor is %d, blk_factor is %d", op_type.c_str(), ub_factor, blk_factor),
                    return false);

  if (ub_axis != blk_axis) {
    return true;
  }
  if (blk_factor%ub_factor == 0) {
    return true;
  }

  float tailPercent = (float)(blk_factor%ub_factor) / (float)ub_factor;
  if (tailPercent >= 0.8) {
    return true;
  }
  int loop = blk_factor / ub_factor + 1;
  ub_factor = blk_factor % loop ? blk_factor / loop + 1 : blk_factor / loop;
  tilingInfo.ub_tiling_factor = ub_factor;

  return true;
}

bool Reduce::Init() {
  try {
    int idx = op_info.count("_idx_before_reduce") > 0 ?
              op_info.at("_idx_before_reduce").get<int>() : 0;
    // CHECK INPUT
    V_OP_TILING_CHECK(!op_paras.inputs.empty(),
                      OP_LOGE(op_type.c_str(), "inputs cannot be empty"),
                      return false);
    V_OP_TILING_CHECK(!(op_paras.inputs.size() <= uint(idx) || idx < 0),
                      OP_LOGE(op_type.c_str(), "idx is invalid index for inputs"),
                      return false);
    V_OP_TILING_CHECK(!op_paras.inputs[uint(idx)].tensor.empty(),
                      OP_LOGE(op_type.c_str(), "tensor cannot be empty"),
                      return false);
    input_shape_ori = op_paras.inputs[idx].tensor[0].shape;

    // Get ori reduce aixs
    if (op_paras.const_inputs.find("axes") != op_paras.const_inputs.end()) {
      // axis_unknown
      TeConstTensorData reduce_axis_info = op_paras.const_inputs.at("axes");
      auto size = std::get<1>(reduce_axis_info);
      ge::DataType axis_type = std::get<2>(reduce_axis_info).GetTensorDesc().GetDataType();
      V_OP_TILING_CHECK(!(axis_type != ge::DT_INT32 && axis_type != ge::DT_INT64),
                        OP_LOGE(op_type.c_str(), "axis_type is not belong to [int32, int64]"),
                        return false);

      if (axis_type == ge::DT_INT32) {
        int count = size / sizeof(int32_t);
        const int32_t* data_addr = reinterpret_cast<const int32_t*>(std::get<0>(reduce_axis_info));
        reduce_axis_ori.resize(count);
        for (int i = 0; i < count; i++) {
          reduce_axis_ori[i] = *data_addr;
          data_addr++;
        }
      } else {
        int count = size / sizeof(int64_t);
        const int64_t* data_addr = reinterpret_cast<const int64_t*>(std::get<0>(reduce_axis_info));
        reduce_axis_ori.resize(count);
        for (int i = 0; i < count; i++) {
          reduce_axis_ori[i] = (int32_t)*data_addr;
          data_addr++;
        }
      }
    } else {
      // axis_known
      reduce_axis_ori = op_info.at("_ori_axis").get<std::vector<int32_t>>();
    }

    // Convert reduce axis (-1 -> length+1)
    // CHECK AXIS VALUE
    int32_t max_value = int32_t(input_shape_ori.size());
    int32_t min_value = -1 * max_value;
    for (size_t i = 0; i < reduce_axis_ori.size(); i++) {
      if (reduce_axis_ori[i] >= max_value || reduce_axis_ori[i] < min_value) {
        OP_LOGE(op_type.c_str(), "value of axis is illegal.");
        return false;
      }
      if (reduce_axis_ori[i] < 0) {
        reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
      }
    }

    // Discard "1" and default sorted
    EliminateOne();
    compileInfo.is_const = op_info.count("_reduce_shape_known") > 0 &&
                           op_info.at("_reduce_shape_known").get<bool>();
  } catch (const std::exception &e) {
    OP_LOGE(op_type.c_str(), "Func of Init error. Error message: %s", e.what());
    return false;
  }

  return true;
}

bool Reduce::WriteTilingData() {
  if (exit_zero_axis) {
    try {
      int status = op_info.at("push_status");
      if (status == 0) {
        ByteBufferPut(run_info.tiling_data, (int32_t)zero_tiling_key);
      }
      ByteBufferPut(run_info.tiling_data, (int32_t)fusion_dim_value);
      ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.ub_tiling_factor);
      run_info.block_dim = (int32_t)1;
      run_info.tiling_key = (int32_t)zero_tiling_key;
    } catch (const std::exception &e) {
      OP_LOGE(op_type.c_str(), "get push status error. Error message: %s", e.what());
      return false;
    }
    return true;
  }

  if (compileInfo.is_const_post) {
    // runtime
    ConstInputProcPost();
    return true;
  }

  if (compileInfo.is_const) {
    // compile
    ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.block_tiling_axis);
    ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.block_tiling_factor);
    ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.ub_tiling_axis);
    ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.ub_tiling_factor);
    run_info.block_dim = tilingInfo.block_dim;
    return true;
  }

  // tiling_key
  run_info.block_dim = tilingInfo.block_dim;
  int32_t tiling_key = CalcTilingKey();
  try {
    run_info.tiling_key = tiling_key;
    int status = op_info.at("push_status");
    if (status == 0) {
      ByteBufferPut(run_info.tiling_data, tiling_key);
    }
  } catch (const std::exception &e) {
    OP_LOGE(op_type.c_str(), "get push_status error. Error message: %s", e.what());
    return false;
  }

  // pure dma_copy, must skip "1".
  uint32_t offset = 0;
  if (input_shape[0] == 1 && reduce_axis[0] == 0) {
    offset = 1;
  }
  for (uint32_t i = offset; i < input_shape.size(); i++) {
    ByteBufferPut(run_info.tiling_data, (int32_t)input_shape[i]);
    OP_LOGD(op_type.c_str(), "input shape:%d", input_shape[i]);
  }

  ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.block_tiling_factor);
  ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.ub_tiling_factor);
  OP_LOGD(op_type.c_str(), "block/res_ub tilling axis:%d", tilingInfo.block_tiling_axis);
  OP_LOGD(op_type.c_str(), "block/res_ub tilling factor:%d", tilingInfo.block_tiling_factor);
  OP_LOGD(op_type.c_str(), "ub/input_ub tilling axis:%d", tilingInfo.ub_tiling_axis);
  OP_LOGD(op_type.c_str(), "ub/input_ub tilling factor:%d", tilingInfo.ub_tiling_factor);

  return true;
}

bool Reduce::DoTiling() {
  /* Situations of DoTiling include:
     1. input(known):
        status of compile: do others except FusedReduceAxis
        status of runtime: do WriteTilingData
     2. input(unknown):
        do all process
  */
  bool ret = true;
  ret = ret && Init();

  for (uint32_t i = 0; i < input_shape_ori.size(); ++i) {
    int64_t dim = input_shape_ori[i];
    bool non_reduce_axis = std::find(reduce_axis_ori.begin(), reduce_axis_ori.end(), i) == reduce_axis_ori.end();

    if (dim == 0) {
      exit_zero_axis = true;
      if (non_reduce_axis) {
        exit_non_reduce_zero_axis = true;
      }
    } else {
      if (non_reduce_axis) {
        fusion_dim_value *= dim;
      }
    }
  }

  if (exit_zero_axis) {
    if (exit_non_reduce_zero_axis) {
      zero_tiling_key = 2147483647;  // EmptySchedule
      tilingInfo.ub_tiling_factor = 128;
    } else {
      try {
        zero_tiling_key = 10;
        tilingInfo.ub_tiling_factor = op_info.at("_zero_ub_factor").get<std::int64_t>();
      } catch (const std::exception &e) {
        OP_LOGE(op_type.c_str(), "get zero_ub_factor error. Error message: %s", e.what());
        return false;
      }
    }

    return true;
  }

  if (compileInfo.is_const) {
    // input(known)
    // invoking the tiling interface during runtime
    // is_const_post: "true"->runtime, "false"->compile
    try {
      compileInfo.is_const_post = op_info.count("_const_shape_post") > 0 &&
              op_info.at("_const_shape_post").get<bool>();
      if (compileInfo.is_const_post) {
        pattern = CalcConstPattern(reduce_axis_ori);
        return ret;
      } else {
        pattern = op_info.at("_compile_pattern").get<std::int32_t>();
        reduce_axis = reduce_axis_ori;
        input_shape = input_shape_ori;
      }
    } catch (const std::exception &e) {
      OP_LOGE(op_type.c_str(), "get compile_info error. Error message: %s", e.what());
      return false;
    }
  } else {
    // input(unknown)
    ret = ret && FusedReduceAxis();
    pattern = CalcPattern(input_shape, reduce_axis);
  }

  // common process
  ret = ret && GetCompileInfo();
  ret = ret && ChooseAtomic();

  if (compileInfo.atomic) {
    run_info.clear_atomic = true;
    ret = ret && ProcessAtomicTiling();
  } else {
    run_info.clear_atomic = false;
    ret = ret && ProcessNormalTiling();
  }
  ret = ret && SpecialUBTiling();
  return ret;
}

bool ReduceTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                  OpRunInfo& run_info) {
  OP_LOGD(op_type.c_str(), "tiling running");
  Reduce reduce(op_type, op_paras, op_info, run_info);
  bool ret = reduce.DoTiling();
  ret = ret && reduce.WriteTilingData();
  return ret;
}
}  // namespace optiling
