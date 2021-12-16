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
 * \file reduce_tiling.cpp
 * \brief tiling function of op
 */
#include "reduce_tiling_v3.h"
#include <algorithm>
#include "error_log.h"
#include "tiling_handler.h"


namespace optiling {
namespace {
  constexpr int32_t SMALL_SHAPE_THRESHOLD = 1024;
  constexpr int32_t FUSED_NON_REDUCE_AXIS = 0;
  constexpr int32_t FUSED_REDUCE_AXIS = 1;
  constexpr int32_t BASE_2 = 2;
  constexpr int32_t BASE_4 = 4;
  constexpr int32_t BASE_10 = 10;
  constexpr int32_t EIGHTY_PERCENT = 0.8;
  constexpr int32_t MAX_INTEGER = 2147483647;
  constexpr int32_t EMPTY_SCHEDULE_UB_TILING_FACTOR_128 = 128;
  constexpr int32_t ARRAY_INDEX_0 = 0;
  constexpr int32_t ARRAY_INDEX_1 = 1;
  constexpr int32_t ARRAY_INDEX_2 = 2;
  constexpr int32_t ARRAY_INDEX_3 = 3;
  constexpr int32_t ARRAY_INDEX_4 = 4;
  constexpr int32_t ARRAY_FIRST_POS = 0;
  constexpr int32_t DEFAULT_CAPACITY_EMPTY = 0;
  constexpr int32_t NO_DIM = 0;
  constexpr int32_t LENGTH_OF_COMMON_FIVE = 5;
  constexpr int32_t TILINGKEY_NONE_REDUCE_AXIS = 2147483646;
}

namespace v3 {
ReduceCompileInfo::ReduceCompileInfo(const std::string& op_type, const nlohmann::json& json_info) {
  bool ret = GetCompileInfoForCalculate(op_type, json_info);
  ret = ret && GetCompileInfoForProcessControl(json_info);
  ret = ret && GetCompileInfoForConst(json_info);
  parsed_success = ret;
}

bool ReduceCompileInfo::GetCompileInfoForProcessControl(const nlohmann::json& json_info) {
  // Optional info from SCH that control the process of tiling
  idx_before_reduce =
          json_info.count("_idx_before_reduce") > 0 ? json_info.at("_idx_before_reduce").get<uint32_t>() : 0;
  is_const = json_info.count("_reduce_shape_known") > 0 && json_info.at("_reduce_shape_known").get<bool>();
  zero_ub_factor = json_info.count("_zero_ub_factor") > 0 ? json_info.at("_zero_ub_factor").get<int64_t>() : -1;
  is_const_post = json_info.count("_const_shape_post") > 0 && json_info.at("_const_shape_post").get<bool>();

  if (json_info.count("_ori_axis") > 0) {
    ori_axis.first = true;
    ori_axis.second = json_info.at("_ori_axis").get<std::vector<int32_t>>();
  }

  if (json_info.count("axes_idx") > 0) {
    axes_idx.first = true;
    axes_idx.second = json_info.at("axes_idx").get<uint32_t>();
  }

  if (json_info.count("_compile_pattern") > 0) {
    compile_pattern.first = true;
    compile_pattern.second = json_info.at("_compile_pattern").get<std::int32_t>();
  }
  return true;
}

bool ReduceCompileInfo::GetCompileInfoForConst(const nlohmann::json& json_info) {
  if (json_info.count("_block_dims") > 0) {
    block_dim_map = json_info.at("_block_dims").get<std::unordered_map<std::string, uint32_t>>();
  }
  if (json_info.count("_atomic_flags") > 0) {
    atomic_flags_map = json_info.at("_atomic_flags").get<std::unordered_map<std::string, bool>>();
  }

  return true;
}

bool ReduceCompileInfo::GetCompileInfoForCalculate(const std::string op_type, const nlohmann::json& json_info) {
  // Required info from SCH that do for calculating
  if (json_info.count("_common_info") > 0) {
    std::vector<int32_t> common_info = json_info.at("_common_info").get<std::vector<int32_t>>();
    if (common_info.size() != LENGTH_OF_COMMON_FIVE) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "_common_info length in json compile info should be 5.");
      return false;
    }
    // Get Data
    core_num = common_info[ARRAY_INDEX_0];
    is_keep_dims = (bool)common_info[ARRAY_INDEX_1];
    min_block_size = common_info[ARRAY_INDEX_2];
    atomic = (bool)common_info[ARRAY_INDEX_3];
    coef = common_info[ARRAY_INDEX_4];
  }

  if (json_info.count("_pattern_info") > 0) {
    pattern_info = json_info.at("_pattern_info").get<std::vector<int32_t>>();
  }

  if (json_info.count("_ub_info_rf") > 0) {
    ub_info_rf = json_info.at("_ub_info_rf").get<std::vector<int32_t>>();
  }

  if (json_info.count("_ub_info") > 0) {
    ub_info = json_info.at("_ub_info").get<std::vector<int32_t>>();
  }

  return true;
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
  int32_t dynamic_pattern = 0;
  for (size_t i = 0; i < input.size(); i++) {
    if (IsInVector(axis, i)) {
      dynamic_pattern += BASE_2 << (input.size() - i - 1);
    } else {
      dynamic_pattern += ((int)input.size() - BASE_2 - (int)i) >= 0 ? BASE_2 << (input.size() - BASE_2 - i) : 1;
    }
  }
  return dynamic_pattern;
}

int32_t Reduce::CalcConstPattern(std::vector<int32_t>& reduce_axis) {
  // generate dict key according to reduce_axis
  // Init() make reduce axis sorted
  if (reduce_axis.size() == 0) {
    return TILINGKEY_NONE_REDUCE_AXIS;
  }

  int32_t dict_key = 0;
  for (auto& i : reduce_axis) {
    // dict_key: 1234 -> reduce [0,1,2,3]
    dict_key = BASE_10 * dict_key + i + 1;
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
        result = result * ((reduceTilingInfo.block_tiling_factor + block_size - 1) / block_size * block_size);
      } else {
        result = result * reduceTilingInfo.block_tiling_factor;
      }
    } else {
      result = result * reorderInfo.reorder_input_shape[i];
    }
  }
  return result;
}

int64_t Reduce::GetAlignShapeMul(int32_t axis_index) const {
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

bool Reduce::CalcBlockDim(std::vector<int64_t>& out, int32_t tiling_axis, int64_t tiling_factor, int32_t& block_dim) {
   if (tiling_factor == 0) {
     VECTOR_INNER_ERR_REPORT_TILIING(op_type, "tiling factor should not be 0");
     return false;
   }

  int32_t block_dim_temp = 1;
  for (int32_t i = 0; i <= tiling_axis; i++) {
    if (out[i] != 0) {
      if (i == tiling_axis) {
        block_dim_temp = (int32_t)((out[i] + tiling_factor - 1) / tiling_factor) * block_dim_temp;
      } else {
        block_dim_temp = (int32_t)out[i] * block_dim_temp;
      }
    }
  }
  block_dim = block_dim_temp;
  return true;
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
  vector<int> pos = {db, shape_type, reduceTilingInfo.block_tiling_axis, reduceTilingInfo.ub_tiling_axis, pattern};
  vector<int> coefficient = {1000000000, 10000000, 1000000, 100000, 100};
  int32_t key = 0;
  for (size_t i = 0; i < coefficient.size(); i++) {
    key += pos[i] * coefficient[i];
  }
  key = reduceTilingInfo.atomic ? key * 1 : -1 * key;
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

bool Reduce::GetConstValue(const ge::GeTensor* geTensor, const ge::DataType& dtype, std::vector<int32_t>& const_vec) {
  size_t size = 0;
  const uint8_t* data_ptr = geTensor->GetData().data();
  V_OP_TILING_CHECK((data_ptr != nullptr),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get nullptr while ge reduce axis from GE."),
                    return false);

  if (dtype == ge::DT_INT32) {
    const void* const_data_ptr_void = static_cast<const void*>(data_ptr);
    const int32_t* const_data_ptr = static_cast<const int32_t*>(const_data_ptr_void);
    size = geTensor->GetData().size() / sizeof(int32_t);
    const_vec.resize(size);
    for (size_t i = 0; i < size; i++) {
      const_vec[i] = (int32_t)(*(const_data_ptr + i));
    }
  } else {
    const void* const_data_ptr_void = static_cast<const void*>(data_ptr);
    const int64_t* const_data_ptr = static_cast<const int64_t*>(const_data_ptr_void);
    size = geTensor->GetData().size() / sizeof(int64_t);
    const_vec.resize(size);
    for (size_t i = 0; i < size; i++) {
      const_vec[i] = static_cast<int32_t>((int64_t)(*(const_data_ptr + i)));
    }
  }
  // clear reduce_axis_ori when shape of input axis is (0,)
  if (size == 0) {
    const_vec.clear();
  }

  return true;
}

void Reduce::FusedReduceAxis() {
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
  size_t first = ARRAY_FIRST_POS;
  size_t second = ARRAY_FIRST_POS;
  int64_t value_input = 0;
  int32_t value_axis = 0;
  size_t length = input_shape_ori.size();
  bool cond_0 = false;
  bool cond_1 = false;

  size_t capacity_shape = DEFAULT_CAPACITY_EMPTY;
  size_t capacity_axis = DEFAULT_CAPACITY_EMPTY;

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
}

void Reduce::ChooseAtomic() {
  // Layer 0 Check if atomic is enabled
  bool atomic_available = compileInfo.atomic;
  // Layer 1 Check if output is large enough (> SMALL_SHAPE_THRESHOLD)
  //         Check normal atomic rules
  reduceTilingInfo.atomic = total_output_count <= ubSizeB &&
                       total_output_count * total_reduce_count > SMALL_SHAPE_THRESHOLD &&
                       total_output_count < static_cast<int64_t>(compileInfo.core_num) * block_size / BASE_2 &&
                       total_reduce_count > static_cast<int64_t>(compileInfo.core_num) / BASE_2;
  // Layer 2 Check if it is nlast_reduce
  //         Check if it is in a0, r, a1 pattern and a0 is 0
  bool is_outermost_nlast_reduce = std::find(reduce_axis.begin(), reduce_axis.end(),
                                             (int32_t)1) != reduce_axis.end() &&
                                             input_shape[0] == (int32_t)1 &&
                                             std::find(reduce_axis.begin(), reduce_axis.end(),
                                                 static_cast<int32_t>(output_shape.size() - 1)) == reduce_axis.end();
  // Check if output_shape is smaller than Single Tensor Size Limitation so that r, a ub_split_a schedule won't be used
  bool output_shape_limitation = total_output_count <= ubSizeB;
  // Check if outermost reduce axis is larger than or equal to core_num
  bool input_shape_limitation = input_shape[1] >= compileInfo.core_num && ubSizeA > SMALL_SHAPE_THRESHOLD * BASE_4;
  // Check nlast_reduce again
  bool n_last_reduce_shape_limitation = ((static_cast<uint32_t>(pattern) & 1) == 1) &&
                                        (input_shape[input_shape.size() - 1] < ubSizeB);
  // AND expression for all checks
  bool shape_limitation = output_shape_limitation && input_shape_limitation && n_last_reduce_shape_limitation;
  // check extracted here because of 120 characters per line static check rule
  reduceTilingInfo.atomic = reduceTilingInfo.atomic || (shape_limitation && is_outermost_nlast_reduce);
  // Final
  reduceTilingInfo.atomic = reduceTilingInfo.atomic && atomic_available;
}

void Reduce::ChooseUBInfo() {
  is_last_axis_reduce = IsInVector(reduce_axis, input_shape.size() - 1);
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

  // UB_SPACE of ub_info is more than ub_info_rf, Atomic selected the former.
  // nodes after reduce(include reduce) have same space(ubSizeA)
  // nodes before reduce have same space(ubSizeB)
  // ubSizeB = ubSizeA * coef (max dtype)
  reduceTilingInfo.max_ub_count = compileInfo.ub_info[reduceTilingInfo.idx];
  ubSizeB = reduceTilingInfo.max_ub_count;
  ubSizeA = ubSizeB / compileInfo.coef;
  block_size = compileInfo.min_block_size;

  // According adaptation of SCH, choose the best UBInfo.
  // Rfactor only attached in Last Reduce.
  if (is_last_axis_reduce) {
    int64_t last_dim = input_shape[input_shape.size()-1];
    int64_t real_reduce_count = total_reduce_count / last_dim;
    last_dim = (last_dim + block_size - 1) / block_size * block_size;
    real_reduce_count *= last_dim;
    bool ub_split_in_r = real_reduce_count > reduceTilingInfo.max_ub_count;
    if (ub_split_in_r) {
      reduceTilingInfo.max_ub_count = compileInfo.ub_info_rf[reduceTilingInfo.idx];
      ubSizeB = reduceTilingInfo.max_ub_count;
      ubSizeA = ubSizeB / compileInfo.coef;
    }
  }
}

bool Reduce::GetUbTilingInfo() {
  // rewrite ub_tiling_factor, ub_tiling_axis
  int32_t block_tiling_axis_in_reorder = -1;
  for (uint32_t i = 0; i < reorderInfo.reorderPos_oriPos.size(); i++) {
    if (reorderInfo.reorderPos_oriPos[i] == reduceTilingInfo.block_tiling_axis) {
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
      reduceTilingInfo.ub_tiling_axis = reorderInfo.reorderPos_oriPos[i];
      reduceTilingInfo.ub_tiling_factor = (ubSizeB / load_mul);

      int64_t max_ub_tiling_factor = input_shape[reduceTilingInfo.ub_tiling_axis];
      if (i == block_tiling_axis_in_reorder) {
        max_ub_tiling_factor = reduceTilingInfo.block_tiling_factor;
      }
      if (reduceTilingInfo.ub_tiling_factor > max_ub_tiling_factor) {
        reduceTilingInfo.ub_tiling_factor = max_ub_tiling_factor;
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
  int32_t block_tiling_axis = reduceTilingInfo.block_tiling_axis;
  int32_t last_reduce_axis_idx = reduce_axis.back();
  reorderInfo.reorder_input_shape.resize(input_shape.size());
  reorderInfo.reorderPos_oriPos.resize(input_shape.size());

  int num_r = reduce_axis.size();
  int num_a = input_shape.size() - num_r;
  for (auto item : reduce_axis) {
    reduce_flag[item] = 1;
  }

  int pos_r = num_a - ((int)input_shape.size() - (last_reduce_axis_idx + 1));
  int pos_a = ARRAY_FIRST_POS;

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
    if (fused_type == FUSED_NON_REDUCE_AXIS && static_cast<int32_t>(i) < block_tiling_axis) {
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
  for (int32_t i = 0; i <= reduceTilingInfo.block_tiling_axis; i++) {
    if (IsInVector(reduce_axis, i)) {
      if (i == reduceTilingInfo.block_tiling_axis) {
        block_dim = (int32_t)((input_shape[i] + reduceTilingInfo.block_tiling_factor - 1) /
                              reduceTilingInfo.block_tiling_factor) * block_dim;
        break;
      } else {
        block_dim = (int32_t)input_shape[i] * block_dim;
      }
    }
  }
  // > 65535 -> false
  reduceTilingInfo.block_dim = block_dim;
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
      reduceTilingInfo.block_tiling_axis = i;
      reduceTilingInfo.block_tiling_factor = 1;

      if (left_mul * input_shape[i] >= core_num) {
        reduceTilingInfo.block_tiling_axis = i;
        int64_t block_tiling_factor_outer = core_num / left_mul;
        reduceTilingInfo.block_tiling_factor = (input_shape[i] + block_tiling_factor_outer - 1) /
                                               block_tiling_factor_outer;
        return is_find_block_tiling;
      }
      left_mul = left_mul * input_shape[i];
    }
  }

  return is_find_block_tiling;
}

void Reduce::GetNotMulCoreBlockTiling() {
  if (input_shape.size() == 0) {
    return;
  }
  reduceTilingInfo.block_tiling_axis = 0;
  reduceTilingInfo.block_tiling_factor = input_shape[0];
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (!IsInVector(reduce_axis, i)) {
      reduceTilingInfo.block_tiling_axis = i;
      reduceTilingInfo.block_tiling_factor = input_shape[i];
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
        if (getBlockTilingInfoX(cur_block_factor, i, right_total_num)) {
          return true;
        }
      } else {
        int64_t cur_block_factor = (block_size + right_total_num - 1) / right_total_num;
        reduceTilingInfo.block_tiling_axis = i;
        reduceTilingInfo.block_tiling_factor = cur_block_factor;
        return true;
      }
    } else if (right_total_num <= block_size || left_block_dim * output_shape[i] >= core_num) {
      if (getBlockTilingInfoY(left_block_dim, i, max_block_tilling_factor, right_total_num)) {
        return true;
      }
    }

    left_block_dim = (int32_t)left_block_dim * output_shape[i];
  }
  return false;
}

bool Reduce::getBlockTilingInfoX(int64_t cur_block_factor, int64_t i, int64_t right_total_num) {
  if (cur_block_factor <= output_shape[i]) {
    for (; cur_block_factor <= output_shape[i]; cur_block_factor++) {
      int64_t tail_block_factor =
          output_shape[i] % cur_block_factor == 0 ? cur_block_factor : output_shape[i] % cur_block_factor;
      int64_t tail_block_tilling_inner_ddr_count = tail_block_factor * right_total_num;
      if (tail_block_tilling_inner_ddr_count >= block_size) {
        reduceTilingInfo.block_tiling_axis = i;
        reduceTilingInfo.block_tiling_factor = cur_block_factor;
        return true;
      }
    }
  } else {
    reduceTilingInfo.block_tiling_axis = i;
    reduceTilingInfo.block_tiling_factor = output_shape[i];
    return true;
  }
  return false;
}

bool Reduce::getBlockTilingInfoY(int32_t left_block_dim, int64_t i,
                                 int64_t max_block_tilling_factor, int64_t right_total_num) {
  int32_t core_num = compileInfo.core_num;
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
                reduceTilingInfo.block_tiling_axis = i;
                reduceTilingInfo.block_tiling_factor = cur_block_factor;
                return true;
              }
            }
          } else {
            reduceTilingInfo.block_tiling_axis = i;
            reduceTilingInfo.block_tiling_factor = cur_block_factor;
            return true;
          }
        } else {
          int64_t block_tilling_inner_ddr_count = cur_block_factor * right_total_num;
          if (block_tilling_inner_ddr_count < block_size) {
            V_OP_TILING_CHECK((right_total_num != 0),
                              VECTOR_INNER_ERR_REPORT_TILIING(op_type, "right_total_num cannot be zero."),
                              return false);
            cur_block_factor = (block_size + right_total_num - 1) / right_total_num;
          }

          reduceTilingInfo.block_tiling_axis = i;
          reduceTilingInfo.block_tiling_factor = cur_block_factor;
          return true;
        }
      }
      return false;
}

bool Reduce::ProcessAtomicTiling() {
  // init
  reduceTilingInfo.block_dim = 0;
  reduceTilingInfo.block_tiling_axis = 0;
  reduceTilingInfo.block_tiling_factor = 0;
  reduceTilingInfo.ub_tiling_axis = 0;
  reduceTilingInfo.ub_tiling_factor = 0;

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
  if (reorderInfo.reorder_input_shape.size() > 0 &&
      reorderInfo.reorder_input_shape.back() % block_size != 0) {
    reorderInfo.reorder_input_shape.back() =
        (reorderInfo.reorder_input_shape.back() + block_size - 1) / block_size * block_size;
  }

  // rewrite TilingInfo(ub)
  return GetUbTilingInfo();
}

bool Reduce::ProcessNormalTiling() {
  // init
  reduceTilingInfo.block_dim = 1;
  reduceTilingInfo.block_tiling_axis = 0;
  reduceTilingInfo.block_tiling_factor = input_shape[0];
  reduceTilingInfo.ub_tiling_axis = 0;
  reduceTilingInfo.ub_tiling_factor = input_shape[0];

  // rewrite TilingInfo(block)
  if (!GetBlockTilingInfo()) {
    if (total_output_count > ubSizeA) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetBlockTilingInfo error");
      return false;
    }
    reduceTilingInfo.block_dim = 1;
    GetNotMulCoreBlockTiling();
  } else {
    if (!CalcBlockDim(output_shape, reduceTilingInfo.block_tiling_axis,
                      reduceTilingInfo.block_tiling_factor, reduceTilingInfo.block_dim)) {
      return false;
    }
  }

  // rewrite ReorderInfo
  ProcessReorderAxis(FUSED_NON_REDUCE_AXIS);
  // align
  if (reorderInfo.reorder_input_shape.size() > 0 &&
      reorderInfo.reorder_input_shape.back() % block_size != 0) {
    reorderInfo.reorder_input_shape.back() =
        (reorderInfo.reorder_input_shape.back() + block_size - 1) / block_size * block_size;
  }

  // rewrite TilingInfo(ub)
  reduceTilingInfo.ub_tiling_axis = reduceTilingInfo.block_tiling_axis;
  reduceTilingInfo.ub_tiling_factor = reduceTilingInfo.block_tiling_factor;
  if (!GetUbTilingInfo()) {
    return false;
  }

  if (!compileInfo.is_keep_dims) {
    reduceTilingInfo.block_tiling_axis = GetRealBlockTilingAxis(output_shape, reduceTilingInfo.block_tiling_axis);
  }

  return true;
}

bool Reduce::FineTuning() {
  /* Fine_tuning for some special case
   * 0. Last Dimensional Segmentation Non-X Alignment, X is 8,16,32...
   * 1. Tail block of ub_factor is too small
   * **/
  int32_t ub_factor = reduceTilingInfo.ub_tiling_factor;
  int32_t blk_factor = reduceTilingInfo.block_tiling_factor;
  int32_t ub_axis = reduceTilingInfo.ub_tiling_axis;
  int32_t blk_axis = reduceTilingInfo.block_tiling_axis;
  int32_t core_num = reduceTilingInfo.block_dim;
  int32_t shape_len = input_shape.size();

  bool pure_data_move = reduce_axis.size() == 1 && input_shape[reduce_axis[0]] == 1;
  if (pure_data_move) {
    return true;
  }

  V_OP_TILING_CHECK(!(ub_factor <= 0 || blk_factor <= 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_factor and blk_factor must bigger than 0,"
                            "while ub_factor is %d, blk_factor is %d", ub_factor, blk_factor),
                    return false);

  bool split_same_dim = ub_axis == blk_axis;
  bool block_split_last_dim = blk_axis == shape_len - 1;
  bool ub_split_last_dim = ub_axis == shape_len - 1;
  bool align_ub_factor = ub_factor % block_size == 0;
  bool align_blk_factor = blk_factor % block_size == 0;
  bool align_last_dim = input_shape[shape_len-1] % block_size == 0;

  // tune_0: branch: split last dim
  // tune_0_0: ub split last_dim
  // tune_0_1: block split last_dim and ub not
  bool tune_0 = (block_split_last_dim || ub_split_last_dim) && align_last_dim;
  bool tune_0_0 = tune_0 && ub_split_last_dim && (not align_ub_factor);
  bool tune_0_1 = tune_0 && block_split_last_dim && (not ub_split_last_dim) && (not align_blk_factor);

  if (tune_0_0) {
    // CoreNum is fixed, ub_factor is maximum
    ub_factor = ub_factor / block_size * block_size;
    reduceTilingInfo.block_tiling_factor = ub_factor > 0 ? ub_factor : reduceTilingInfo.block_tiling_factor;
    return true;
  }

  if (tune_0_1) {
    // blk_factor can be upper or lower to block_size * n
    // Regulation: upper -> lower -> abandon
    int32_t upper_value = (blk_factor + block_size - 1) / block_size * block_size;
    int32_t lower_value = blk_factor / block_size * block_size;
    int32_t const_value = blk_factor * ub_factor;
    core_num = core_num / ((input_shape[blk_axis] + blk_factor - 1) / blk_factor);

    if (const_value / upper_value > 0 && upper_value <= input_shape[blk_axis] && upper_value <= ubSizeA) {
      // core_num = A*B*Blk_outer
      core_num = core_num * ((input_shape[blk_axis] + upper_value - 1) / upper_value);
      reduceTilingInfo.block_dim = core_num;
      reduceTilingInfo.block_tiling_factor = upper_value;
      reduceTilingInfo.ub_tiling_factor = const_value / upper_value;
    } else if (lower_value > 0) {
      core_num = core_num * ((input_shape[blk_axis] + lower_value - 1) / lower_value);
      reduceTilingInfo.block_dim = core_num;
      reduceTilingInfo.block_tiling_factor = lower_value;
      int32_t expect_value = const_value / lower_value;
      reduceTilingInfo.ub_tiling_factor = expect_value >= input_shape[ub_axis] ? input_shape[ub_axis] : expect_value;
    }
    return true;
  }

  // tune_1
  bool tune_1 = split_same_dim && (blk_factor % ub_factor != 0);
  if (tune_1) {
    float tailPercent = static_cast<float>(blk_factor % ub_factor) / static_cast<float>(ub_factor);
    if (tailPercent >= EIGHTY_PERCENT) {
      return true;
    }
    int loop = blk_factor / ub_factor + 1;
    ub_factor = (blk_factor % loop) ? blk_factor / loop + 1 : blk_factor / loop;
    reduceTilingInfo.ub_tiling_factor = ub_factor;
    return true;
  }

  return true;
}

bool Reduce::IsZero() {
  for (uint32_t i = 0; i < input_shape_ori.size(); ++i) {
    int64_t dim = input_shape_ori[i];
    bool non_reduce_axis = std::find(reduce_axis_ori.begin(), reduce_axis_ori.end(), i) == reduce_axis_ori.end();

    if (dim == NO_DIM) {
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
  return exit_zero_axis;
}

bool Reduce::DoZeroBranch() {
  if (exit_non_reduce_zero_axis) {
    // EmptySchedule
    zero_tiling_key = MAX_INTEGER;
    reduceTilingInfo.ub_tiling_factor = EMPTY_SCHEDULE_UB_TILING_FACTOR_128;
  } else {
    zero_tiling_key = BASE_10;
    reduceTilingInfo.ub_tiling_factor = compileInfo.zero_ub_factor;
    V_OP_TILING_CHECK(reduceTilingInfo.ub_tiling_factor > 0,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                      "zero_ub_factor is %ld, which is init_value",
                      reduceTilingInfo.ub_tiling_factor),
                      return false);
  }
  return true;
}

bool Reduce::DoConstRunTimeBranch() {
  // runtime in DoTiling while input is const
  EliminateOne();
  pattern = CalcConstPattern(reduce_axis_ori);
  std::string pattern_str = std::to_string(pattern);
  reduceTilingInfo.const_block_dims = compileInfo.block_dim_map.at(pattern_str);
  reduceTilingInfo.const_atomic_flag = compileInfo.atomic_flags_map.at(pattern_str);
  return true;
}

bool Reduce::SetAttrVars(int32_t key) const {
  return true;
}

bool Reduce::WriteTilingData() {
  if (reduceTilingInfo.atomic) {
    run_info.SetClearAtomic(true);
  } else {
    run_info.SetClearAtomic(false);
  }

  if (exit_zero_axis) {
    run_info.AddTilingData((int32_t)fusion_dim_value);
    run_info.AddTilingData((int32_t)reduceTilingInfo.ub_tiling_factor);
    run_info.SetBlockDim(1);
    run_info.SetTilingKey(static_cast<uint32_t>(zero_tiling_key));
    return SetAttrVars(zero_tiling_key);
  }

  if (compileInfo.is_const_post) {
    // runtime
    run_info.SetBlockDim(reduceTilingInfo.const_block_dims);
    run_info.SetClearAtomic(reduceTilingInfo.const_atomic_flag);
    run_info.SetTilingKey(static_cast<uint32_t>(pattern));
    return SetAttrVars(pattern);
  }

  if (compileInfo.is_const) {
    // compile
    run_info.AddTilingData((int32_t)reduceTilingInfo.block_tiling_axis);
    run_info.AddTilingData((int32_t)reduceTilingInfo.block_tiling_factor);
    run_info.AddTilingData((int32_t)reduceTilingInfo.ub_tiling_axis);
    run_info.AddTilingData((int32_t)reduceTilingInfo.ub_tiling_factor);
    run_info.AddTilingData(static_cast<int32_t>(ubSizeB));
    run_info.AddTilingData(static_cast<int32_t>(ubSizeA));
    run_info.SetBlockDim(static_cast<uint32_t>(reduceTilingInfo.block_dim));
    return true;
  }

  // tiling_key
  run_info.SetBlockDim(static_cast<uint32_t>(reduceTilingInfo.block_dim));
  int32_t tiling_key = CalcTilingKey();
  run_info.SetTilingKey(static_cast<uint32_t>(tiling_key));

  // pure dma_copy, must skip "1".
  uint32_t offset = 0;
  if (input_shape[0] == 1 && reduce_axis[0] == 0) {
    offset = 1;
  }
  for (uint32_t i = offset; i < input_shape.size(); i++) {
    run_info.AddTilingData((int32_t)input_shape[i]);
    OP_LOGD(op_type.c_str(), "input shape:%d", input_shape[i]);
  }

  run_info.AddTilingData((int32_t)reduceTilingInfo.block_tiling_factor);
  run_info.AddTilingData((int32_t)reduceTilingInfo.ub_tiling_factor);
  OP_LOGD(op_type.c_str(), "block/res_ub tilling axis:%d", reduceTilingInfo.block_tiling_axis);
  OP_LOGD(op_type.c_str(), "block/res_ub tilling factor:%d", reduceTilingInfo.block_tiling_factor);
  OP_LOGD(op_type.c_str(), "ub/input_ub tilling axis:%d", reduceTilingInfo.ub_tiling_axis);
  OP_LOGD(op_type.c_str(), "ub/input_ub tilling factor:%d", reduceTilingInfo.ub_tiling_factor);

  return SetAttrVars(tiling_key);
}

bool Reduce::CheckCompileInfoForCalculate() {
  // Required info from SCH that do for calculating
  V_CHECK_EQ(compileInfo.pattern_info.size(), compileInfo.ub_info.size(),
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "pattern_info's size should be as same as ub_info"),
               return false);
  V_CHECK_EQ(compileInfo.pattern_info.size(), compileInfo.ub_info_rf.size(),
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "pattern_info's size should be as same as ub_info_rf"),
               return false);

  // CHECK VALUE
  V_OP_TILING_CHECK(!(compileInfo.coef <= 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "coef is %d that is illegal", compileInfo.coef),
                      return false);
  V_OP_TILING_CHECK(!(compileInfo.min_block_size <= 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                      "min_block_size is %d that is illegal",
                                                      compileInfo.min_block_size),
                      return false);
  V_OP_TILING_CHECK(!(compileInfo.core_num <= 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "core_num is %d that is illegal", compileInfo.core_num),
                      return false);
  return true;
}

bool Reduce::GetGeInfo() {
  // Get Input
  uint32_t inputs_num = op_paras.GetInputsSize();
  V_OP_TILING_CHECK(!(inputs_num == 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "inputs cannot be empty"),
                    return false);
  V_OP_TILING_CHECK(!(inputs_num <= compileInfo.idx_before_reduce),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "idx is invalid index for inputs"),
                    return false);
  input_shape_ori = ge::OpDescUtils::GetOpDescFromOperator(
      op_paras)->MutableInputDesc(compileInfo.idx_before_reduce)->GetShape().GetDims();

  return true;
}

bool Reduce::SetInit() {
  // Get ReduceAxis
  bool ret = getReduceAxisTensor();
  if (!ret) {
    return false;
  }
  // Convert reduce axis (-1 -> length+1)
  int32_t max_value = int32_t(input_shape_ori.size());
  int32_t min_value = -1 * max_value;
  for (size_t i = 0; i < reduce_axis_ori.size(); i++) {
    if (reduce_axis_ori[i] >= max_value || reduce_axis_ori[i] < min_value) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of axis is %d which exceed max_value: %d or min_value: %d",
                                      reduce_axis_ori[i], max_value, min_value);
      return false;
    }
    if (reduce_axis_ori[i] < 0) {
      reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
    }
  }
  return true;
}

bool Reduce::MatchPattern() {
  for (auto item: compileInfo.pattern_info) {
    if (item == pattern) {
      break;
    }
    reduceTilingInfo.idx += 1;
  }

  // CHECK VALUE
  if (reduceTilingInfo.idx >= compileInfo.pattern_info.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "pattern is %d that not in pattern_info", pattern);
    return false;
  }
  // Use compileInfo.idx to choose buffer's size
  ChooseUBInfo();
  ChooseAtomic();
  return true;
}

bool Reduce::TilingProcess() {
  if (reduceTilingInfo.atomic) {
    return ProcessAtomicTiling();
  } else {
    return ProcessNormalTiling();
  }
}

bool Reduce::DoReduceTiling() {
  /* Situations of DoTiling include:
     1. input(known):
        status of compile: do others except FusedReduceAxis
        status of runtime: do WriteTilingData
     2. input(unknown):
        do all process
  */
  V_OP_TILING_CHECK(GetGeInfo(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetGeInfo Failed"),
                    return false);
  V_OP_TILING_CHECK(SetInit(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SetInit Failed"),
                    return false);

  if (IsZero()){
    return DoZeroBranch();
  }

  if (compileInfo.is_const && compileInfo.is_const_post) {
    // status: runtime
    return DoConstRunTimeBranch();
  }

  if (compileInfo.is_const) {
    // status: compile
    V_OP_TILING_CHECK(compileInfo.compile_pattern.first,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get compile_pattern Failed"),
                      return false);
    pattern = compileInfo.compile_pattern.second;
    reduce_axis = reduce_axis_ori;
    input_shape = input_shape_ori;
  } else {
    // input(unknown)
    // Discard "1" and default sorted
    EliminateOne();
    FusedReduceAxis();
    pattern = CalcPattern(input_shape, reduce_axis);
  }

  // common process
  V_OP_TILING_CHECK(CheckCompileInfoForCalculate(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CheckCompileInfoForCalculate Failed"),
                    return false);
  V_OP_TILING_CHECK(MatchPattern(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "MatchPattern Failed"),
                    return false);
  V_OP_TILING_CHECK(TilingProcess(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TilingProcess Failed"),
                    return false);
  V_OP_TILING_CHECK(FineTuning(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "FineTuning Failed"),
                    return false);

  return true;
}

bool Reduce::getReduceAxisTensor() {
  // Get ReduceAxis
  if (compileInfo.ori_axis.first) {
    reduce_axis_ori = compileInfo.ori_axis.second;
  } else {
    // axes is tensor
    // te_fusion will convert axes_idx while do fusion in reduce_op
    uint32_t axes_idx = compileInfo.axes_idx.first ? compileInfo.axes_idx.second:
                   ge::OpDescUtils::GetOpDescFromOperator(op_paras)->GetInputIndexByName("axes");
    auto axes_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(axes_idx);
    ge::DataType axes_type = axes_desc->GetDataType();
    const ge::GeTensor* axis_tensor = ge::OpDescUtils::GetInputConstData(op_paras, axes_idx);

    // push data in axis_tensor
    V_OP_TILING_CHECK(axis_tensor != nullptr,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetInputConstData Failed"),
                      return false);
    V_OP_TILING_CHECK(!(axes_type != ge::DT_INT32 && axes_type != ge::DT_INT64),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "axes_type is %d, not belong to [int32, int64]",
                                                      axes_type),
                      return false);
    if (!GetConstValue(axis_tensor, axes_type, reduce_axis_ori)) {
      return false;
    }
  }

  return true;
}


bool Reduce::DoReduceTiling(const OpInfo& op_info) {
  /* Situations of DoTiling include:
     1. input(known):
        status of compile: do others except FusedReduceAxis
        status of runtime: do WriteTilingData
     2. input(unknown):
        do all process
  */
  // Get Input
  if (op_info.GetInputShape().size() > 0) {
    input_shape_ori = op_info.GetInputShape()[0];
  } else {
    uint32_t inputs_num = op_paras.GetInputsSize();
    V_OP_TILING_CHECK(!(inputs_num == 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "inputs cannot be empty"),
                      return false);
    V_OP_TILING_CHECK(!(inputs_num <= compileInfo.idx_before_reduce),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "idx is invalid index for inputs"),
                      return false);
    input_shape_ori = ge::OpDescUtils::GetOpDescFromOperator(
      op_paras)->MutableInputDesc(compileInfo.idx_before_reduce)->GetShape().GetDims();
  }

  if (op_info.GetReduceAxes().size() > 0) {
    reduce_axis_ori = op_info.GetReduceAxes()[0];
  } else {
    bool ret = getReduceAxisTensor();
    if (!ret) {
      return false;
    }
  }

  // Convert reduce axis (-1 -> length+1)
  int32_t max_value = int32_t(input_shape_ori.size());
  int32_t min_value = -1 * max_value;
  for (size_t i = 0; i < reduce_axis_ori.size(); i++) {
    if (reduce_axis_ori[i] >= max_value || reduce_axis_ori[i] < min_value) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of axis is %d which exceed max_value: %d or min_value: %d",
                                      reduce_axis_ori[i], max_value, min_value);
      return false;
    }
    if (reduce_axis_ori[i] < 0) {
      reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
    }
  }

  if (IsZero()){
    return DoZeroBranch();
  }

  if (compileInfo.is_const && compileInfo.is_const_post) {
    // status: runtime
    return DoConstRunTimeBranch();
  }

  if (compileInfo.is_const) {
    // status: compile
    V_OP_TILING_CHECK(compileInfo.compile_pattern.first,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get compile_pattern Failed"), return false);
    pattern = compileInfo.compile_pattern.second;
    reduce_axis = reduce_axis_ori;
    input_shape = input_shape_ori;
  } else {
    // input(unknown)
    // Discard "1" and default sorted
    EliminateOne();
    FusedReduceAxis();
    pattern = CalcPattern(input_shape, reduce_axis);
  }

  // common process
  V_OP_TILING_CHECK(CheckCompileInfoForCalculate(), VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                             "CheckCompileInfoForCalculate Failed"), return false);
  V_OP_TILING_CHECK(MatchPattern(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "MatchPattern Failed"), return false);
  V_OP_TILING_CHECK(TilingProcess(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TilingProcess Failed"), return false);
  V_OP_TILING_CHECK(FineTuning(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "FineTuning Failed"), return false);

  return true;
}

bool Reduce::DoTiling() {
  bool ret = DoReduceTiling();
  ret = ret && WriteTilingData();
  return ret;
}

bool Reduce::DoTiling(const OpInfo& op_info) {
  bool ret = DoReduceTiling(op_info);
  ret = ret && WriteTilingData();
  return ret;
}
} // namespace v3

bool ReduceTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "tiling running");
  v3::Reduce reduce(op_type, op_paras, compileInfo, run_info);
  return reduce.DoTiling();
}

bool ReduceTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                   const OpInfo& op_info) const {
  OP_LOGD(op_type.c_str(), "user-defined shape tiling running");
  v3::Reduce reduce(op_type, op_paras, compileInfo, run_info);
  return reduce.DoTiling(op_info);
}

std::shared_ptr<AutoTilingHandler> CreateReduceTilingHandler(const std::string& op_type,
                                                             const std::string& pattern,
                                                             const nlohmann::json& parsed_compile_info) {
  auto reduceCompileInfoV3_ptr = std::make_shared<ReduceTilingHandler>(op_type, pattern, parsed_compile_info);

  return reduceCompileInfoV3_ptr->ParsedSuccess() ?
                        reduceCompileInfoV3_ptr : std::shared_ptr<AutoTilingHandler>(nullptr);
}
}  // namespace optiling
