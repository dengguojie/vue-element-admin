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

#include "reduce_tiling.h"

namespace optiling {

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

int32_t Reduce::GetBlockSize(std::string dtypeUB) {
  int32_t block_size = 0;

  if (dtypeUB == "float32" || dtypeUB == "int32" || dtypeUB == "uint32") {
    block_size = 8;
  } else if (dtypeUB == "float16" || dtypeUB == "int16" || dtypeUB == "uint16") {
    block_size = 16;
  } else if (dtypeUB == "int8" || dtypeUB == "uint8") {
    block_size = 32;
  }

  return block_size;
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
}

bool Reduce::ConstInputProcPost() {
  // runtime
  std::string pattern_str = std::to_string(pattern);
  run_info.block_dim = op_info["_block_dims"][pattern_str].get<std::int32_t>();
  run_info.clear_atomic = op_info["_atomic_flags"][pattern_str].get<bool>();
  ByteBufferPut(run_info.tiling_data, pattern);
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
  std::vector<int32_t> info = op_info["common_info"];
  compileInfo.max_ub_count = info[0];
  compileInfo.core_num = info[1];
  compileInfo.is_keep_dims = (bool)info[2];
  compileInfo.reduce_block_size = info[3];
  compileInfo.atomic = (bool)info[4];
  output_dtypeUB = op_paras.inputs[0].tensor[0].dtype;
  is_last_axis_reduce = IsInVector(reduce_axis, input_shape.size() - 1);

  // special process(reduce_mean_d)
  set_reduce_mean_cof_flag = 0;
  // reduce_mean_cof is not required when handling pure dma_copy case
  if (input_shape[0] == 1 && reduce_axis[0] == 0) {
    return true;
  }
  reduce_mean_cof = 1.0;
  if (op_info.count("reduce_mean_cof_dtype") > 0) {
    const std::string& reduce_mean_cof_dtype = op_info["reduce_mean_cof_dtype"].get<std::string>();
    if (reduce_mean_cof_dtype == "float32") {
      set_reduce_mean_cof_flag = REDUCE_MEAN_COF_FP32;
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (IsInVector(reduce_axis, i)) {
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        }
      }
    } else if (reduce_mean_cof_dtype == "float16") {
      set_reduce_mean_cof_flag = REDUCE_MEAN_COF_FP16;
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (IsInVector(reduce_axis, i)) {
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        }
      }
    }
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

  int32_t calc_block_size = compileInfo.reduce_block_size;
  int32_t output_block_size = GetBlockSize(output_dtypeUB);
  block_size = calc_block_size > output_block_size ? calc_block_size : output_block_size;

  compileInfo.atomic = compileInfo.atomic && total_output_count <= compileInfo.max_ub_count &&
                       total_output_count * total_reduce_count > SMALL_SHAPE_THRESHOLD &&
                       total_output_count < (int64_t)compileInfo.core_num * block_size / 2 &&
                       total_reduce_count > (int64_t)compileInfo.core_num / 2;
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
    if (load_mul <= compileInfo.max_ub_count) {
      tilingInfo.ub_tiling_axis = reorderInfo.reorderPos_oriPos[i];
      tilingInfo.ub_tiling_factor = (compileInfo.max_ub_count / load_mul);

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
  int32_t reorder_pos = 0;
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
  for (size_t i = 0; i <= last_reduce_axis_idx; i++) {
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
    if (fused_type == FUSED_NON_REDUCE_AXIS && i < block_tiling_axis) {
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
  int64_t max_ub_count = compileInfo.max_ub_count;
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
    if (total_output_count > compileInfo.max_ub_count) {
      GE_LOGE("GetBlockTilingInfo error.");
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

bool Reduce::Init() {
  // get ori input_shape
  if (op_paras.inputs.size() > 0 && op_paras.inputs[0].tensor.size() > 0) {
    input_shape_ori = op_paras.inputs[0].tensor[0].shape;
  } else {
    GE_LOGE("op [%s] : input shape error.", op_type.c_str());
    return false;
  }

  // get ori reduce axis
  if (op_paras.const_inputs.find("axes") != op_paras.const_inputs.end()) {
    // axis_unknown
    const std::string& axis_type = op_paras.inputs[1].tensor[0].dtype;
    TeConstTensorData reduce_axis_info = op_paras.const_inputs.at("axes");
    auto size = std::get<1>(reduce_axis_info);
    if (axis_type == "int32") {
      int count = size / sizeof(int32_t);
      const int32_t* data_addr = reinterpret_cast<const int32_t*>(std::get<0>(reduce_axis_info));
      reduce_axis_ori.resize(count);
      for (int i = 0; i < count; i++) {
        reduce_axis_ori[i] = *data_addr;
        data_addr++;
      }
    } else if (axis_type == "int64") {
      int count = size / sizeof(int64_t);
      const int64_t* data_addr = reinterpret_cast<const int64_t*>(std::get<0>(reduce_axis_info));
      reduce_axis_ori.resize(count);
      for (int i = 0; i < count; i++) {
        reduce_axis_ori[i] = (int32_t)*data_addr;
        data_addr++;
      }
    } else {
      GE_LOGE("op [%s] : axis type is not int32 or int64.", op_type.c_str());
      return false;
    }
  } else {
    // axis_known
    const auto& reduce_axis_tmp = op_info["_ori_axis"];
    reduce_axis_ori.resize(reduce_axis_tmp.size());
    size_t i = 0;
    for (const auto& axis : reduce_axis_tmp) {
      reduce_axis_ori[i] = axis;
      i++;
    }
  }

  // convert reduce axis (-1 -> length+1)
  for (size_t i = 0; i < reduce_axis_ori.size(); i++) {
    if (reduce_axis_ori[i] < 0) {
      reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
    }
  }

  // discard "1" and default sorted
  EliminateOne();
  compileInfo.is_const = op_info.count("reduce_shape_known") > 0 && op_info["reduce_shape_known"].get<bool>();

  return true;
}

bool Reduce::WriteTilingData() {
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
  ByteBufferPut(run_info.tiling_data, tiling_key);

  // pure dma_copy, must skip "1".
  uint32_t offset = 0;
  if (input_shape[0] == 1 && reduce_axis[0] == 0) {
    offset = 1;
  }
  for (uint32_t i = offset; i < input_shape.size(); i++) {
    ByteBufferPut(run_info.tiling_data, (int32_t)input_shape[i]);
    GELOGD("input shape:%d", input_shape[i]);
  }

  if (set_reduce_mean_cof_flag == REDUCE_MEAN_COF_FP32) {
    ByteBufferPut(run_info.tiling_data, (float)reduce_mean_cof);
    GELOGD("reduce mean cof:%f", reduce_mean_cof);
  } else if (set_reduce_mean_cof_flag == REDUCE_MEAN_COF_FP16) {
    fe::fp16_t reduce_mean_cof_fp16;
    reduce_mean_cof_fp16 = reduce_mean_cof;
    ByteBufferPut(run_info.tiling_data, (fe::fp16_t)reduce_mean_cof_fp16);
    ByteBufferPut(run_info.tiling_data, (uint16_t)0);
    GELOGD("reduce mean cof:%f", reduce_mean_cof);
  }

  ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.block_tiling_factor);
  ByteBufferPut(run_info.tiling_data, (int32_t)tilingInfo.ub_tiling_factor);
  GELOGD("block/res_ub tilling axis:%d", tilingInfo.block_tiling_axis);
  GELOGD("block/res_ub tilling factor:%d", tilingInfo.block_tiling_factor);
  GELOGD("ub/input_ub tilling axis:%d", tilingInfo.ub_tiling_axis);
  GELOGD("ub/input_ub tilling factor:%d", tilingInfo.ub_tiling_factor);

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
  GELOGD("op [%s]: tiling running", op_type.c_str());
  bool ret = true;
  ret = ret && Init();

  if (compileInfo.is_const) {
    // input(known)
    // invoking the tiling interface during runtime
    // is_const_post: "true"->runtime, "false"->compile
    pattern = CalcConstPattern(reduce_axis_ori);
    compileInfo.is_const_post = op_info.count("const_shape_post") > 0 && op_info["const_shape_post"].get<bool>();
    if (compileInfo.is_const_post) {
      return ret;
    } else {
      reduce_axis = reduce_axis_ori;
      input_shape = input_shape_ori;
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

  return ret;
}

bool ReduceTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                  OpRunInfo& run_info) {
  Reduce reduce(op_type, op_paras, op_info, run_info);
  bool ret = reduce.DoTiling();
  ret = ret && reduce.WriteTilingData();
  return ret;
}
}  // namespace optiling
