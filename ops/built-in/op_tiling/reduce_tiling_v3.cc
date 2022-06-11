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
#include "auto_tiling_register.h"


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
  constexpr int32_t ARRAY_INDEX_5 = 5;
  constexpr int32_t ARRAY_INDEX_6 = 6;
  constexpr int32_t ARRAY_FIRST_POS = 0;
  constexpr int32_t DEFAULT_CAPACITY_EMPTY = 0;
  constexpr int32_t NO_DIM = 0;
  constexpr int32_t LEAST_LENGTH_OF_COMMON_FIVE = 5;
  constexpr int32_t LENGTH_OF_COMMON_SIX = 6;
  constexpr int32_t LENGTH_OF_COMMON_SEVEN = 7;
  constexpr int32_t LENGTH_OF_COMMON_EIGHT = 8;
  constexpr int32_t LENGTH_OF_COMMON_NINE = 9;
  constexpr int32_t TILINGKEY_NONE_REDUCE_AXIS = 2147483646;
  constexpr int32_t MIN_NOT_ONE_AXIS_NUM = 2;
  constexpr int32_t SHAPE_LENGTH_TWO = 2;
  constexpr int32_t SHAPE_LENGTH_THREE = 3;
  constexpr int32_t INDEX_OF_LAST_DIM_OF_ARA_CASE = 2;
  constexpr int32_t REDUCE_PAD_SCH_TYPE = 1;
  constexpr int32_t REDUCE_TRANSPOSE_SCH_TYPE = 2;
  constexpr int32_t TRANSPOSE_THRESHOLD_VALUE = 64;
  constexpr int32_t REDUCE_AXES_TYPE_ALL = 0;
  constexpr int32_t REDUCE_PRODUCT_COEFFICIENT = 64;
  constexpr int32_t FAKE_WORKSPACE_SIZE = 32;
  constexpr int32_t ALIGN_BYTES = 32;
  // each core need 32 + 4 int64_t size
  constexpr int32_t SINGLE_SYNC_CORE_BYTES = 288;
  constexpr int32_t REDUCE_MAX_WORKSPACE_NUMS = 20;
  constexpr int32_t MODULO = 10;
}

namespace v3 {
ReduceCompileInfo::ReduceCompileInfo(const char* op_type, const nlohmann::json& json_info) {
  parsed_success = Parse(op_type, json_info);
}

bool ReduceCompileInfo::Parse(const char* op_type, const nlohmann::json& json_info) {
  bool ret = GetCompileInfoForCalculate(op_type, json_info);
  ret = ret && GetCompileInfoForProcessControl(json_info);
  ret = ret && GetCompileInfoForConst(json_info);
  ret = ret && GetCompileInfoForRunInfo(json_info);
  return ret;
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
    ori_axis.second = json_info.at("_ori_axis").get<std::vector<int64_t>>();
  }

  if (json_info.count("axes_idx") > 0) {
    axes_idx.first = true;
    axes_idx.second = json_info.at("axes_idx").get<uint32_t>();
  }

  if (json_info.count("_compile_pattern") > 0) {
    compile_pattern.first = true;
    compile_pattern.second = json_info.at("_compile_pattern").get<std::int32_t>();
  }

  if (json_info.count("_reduce_axes_type") > 0) {
    reduce_axes_type = json_info.at("_reduce_axes_type").get<std::int32_t>();
  }

  if (json_info.count("_workspace_size") > 0) {
    workspace_size = json_info.at("_workspace_size").get<std::int32_t>();
  }
  
  if(json_info.count("_reduce_vars") > 0) {
    const auto& local_reduce_vars =
      json_info.at("_reduce_vars").get<std::unordered_map<std::string, std::vector<int32_t>>>();
    for (const auto& single_item: local_reduce_vars) {
      reduce_vars[std::stoull(single_item.first)] = single_item.second;
    }
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

bool ReduceCompileInfo::GetCompileInfoForCalculate(const char* op_type, const nlohmann::json& json_info) {
  // Required info from SCH that do for calculating
  if (json_info.count("_common_info") > 0) {
    if(!GetCompileCommonInfo(op_type, json_info)) {
      return false;
    }
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

  if (json_info.count("_ub_info_pad") > 0) {
    ub_info_pad = json_info.at("_ub_info_pad").get<std::vector<int32_t>>();
  }

  if (json_info.count("_ub_info_transpose") > 0) {
    ub_info_transpose = json_info.at("_ub_info_transpose").get<std::vector<int32_t>>();
  }

  if (json_info.count("_disable_fuse_axes") > 0) {
    disable_fuse_axes = json_info.at("_disable_fuse_axes").get<std::vector<int32_t>>();
  }

  ori_dim_index =
        json_info.count("_ori_dim_index") > 0 ? json_info.at("_ori_dim_index").get<uint32_t>() : -1;

  return true;
}

bool ReduceCompileInfo::GetCompileCommonInfo(const char* op_type, const nlohmann::json& json_info) {
  std::vector<int32_t> common_info = json_info.at("_common_info").get<std::vector<int32_t>>();
  if (common_info.size() < LEAST_LENGTH_OF_COMMON_FIVE) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "_common_info length in json compile info should be at least 5.");
    return false;
  }
  // Get Data
  core_num = common_info[ARRAY_INDEX_0];
  is_keep_dims = (bool)common_info[ARRAY_INDEX_1];
  min_block_size = common_info[ARRAY_INDEX_2];
  atomic = (bool)common_info[ARRAY_INDEX_3];
  coef = common_info[ARRAY_INDEX_4];

  if (common_info.size() >= LENGTH_OF_COMMON_SIX) {
    pad_max_entire_size = common_info[ARRAY_INDEX_5];
    if (pad_max_entire_size <= 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "pad max entire size is %d that is illegal.", pad_max_entire_size);
      return false;
    }
  }

  if (common_info.size() >= LENGTH_OF_COMMON_SEVEN) {
    support_transpose = (bool)common_info[ARRAY_INDEX_6];
  }

  if (common_info.size() >= LENGTH_OF_COMMON_EIGHT) {
    reduce_dtype_byte = common_info[LENGTH_OF_COMMON_EIGHT - 1];
    if (reduce_dtype_byte <= 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "reduce dtype byte is %d that is illegal.", reduce_dtype_byte);
      return false;
    }
  }

  if (common_info.size() >= LENGTH_OF_COMMON_NINE) {
    group_reduce = (bool)common_info[LENGTH_OF_COMMON_NINE - 1];
  }

  return true;
}

bool ReduceCompileInfo::GetCompileInfoForRunInfo(const nlohmann::json& json_info) {
  return var_attr_wrap.ParseVarAttr(json_info);
}

template <typename T>
bool Reduce<T>::IsInVector(std::vector<int64_t>& input, int32_t value) {
  for (uint32_t i = 0; i < input.size(); i++) {
    if (input[i] == value) {
      return true;
    }
  }
  return false;
}

template <typename T>
int32_t Reduce<T>::CalcPattern(std::vector<int64_t>& input, std::vector<int64_t>& axis) {
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

template <typename T>
int32_t Reduce<T>::CalcConstPattern(std::vector<int64_t>& reduce_axis) {
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

template <typename T>
int64_t Reduce<T>::GetReorderInputShapeMul(int32_t axis_index, int32_t block_tiling_axis_in_reorder) {
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

template <typename T>
int64_t Reduce<T>::GetAlignShapeMul(int32_t axis_index) const {
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

template <typename T>
int64_t Reduce<T>::GetShapeMul(std::vector<int64_t>& shape, int32_t axis_index) {
  int64_t result = 1;
  for (uint32_t i = axis_index + 1; i < shape.size(); i++) {
    if (shape[i] != 0) {
      result = result * shape[i];
    }
  }
  return result;
}

template <typename T>
bool Reduce<T>::CalcBlockDim(std::vector<int64_t>& out, int32_t tiling_axis,
                             int64_t tiling_factor, int32_t& block_dim) {
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

template <typename T>
int32_t Reduce<T>::GetRealBlockTilingAxis(std::vector<int64_t>& shape, int32_t idx) {
  int32_t zero_cnt = 0;
  for (int32_t i = 0; i < idx; i++) {
    if (shape[i] == 0) {
      zero_cnt += 1;
    }
  }
  return idx - zero_cnt;
}

template <typename T>
int32_t Reduce<T>::CalcTilingKey() {
  using namespace std;
  int db = 0;
  vector<int> pos = {db, reduceTilingInfo.sch_type, reduceTilingInfo.block_tiling_axis,
                     reduceTilingInfo.ub_tiling_axis, pattern};
  vector<int> coefficient = {1000000000, 10000000, 1000000, 100000, 100};
  int32_t key = 0;
  for (size_t i = 0; i < coefficient.size(); i++) {
    key += pos[i] * coefficient[i];
  }
  key = reduceTilingInfo.atomic ? key * 1 : -1 * key;
  return key;
}

template <typename T>
void Reduce<T>::EliminateOne() {
  // not elminate one if is 5hd case
  if (!compileInfo->disable_fuse_axes.empty()) {
    return;
  }
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

template <typename T>
void Reduce<T>::FusedReduceAxis() {
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

  // deal 5hd case
  if (!compileInfo->disable_fuse_axes.empty()) {
     DealDisableFuseAxes(pos);
  }
  while (second <= length) {
    if (second <= length - 1 and pos[first] == pos[second]) {
      // look for unequal idx
      second += 1;
    } else {
      // fused serial axises
      value_input = std::accumulate(input_shape_ori.begin() + first, input_shape_ori.begin() + second, 1,
                                    std::multiplies<int64_t>());
      input_shape[capacity_shape] = value_input;

      // cond_0: [first, second) is serial reduce_axises
      // cond_1: [first: ] is serial reduce_axises.
      cond_0 = second <= length - 1 and pos[first] % MODULO == 1;
      cond_1 = second == length and pos[second - 1] % MODULO == 1;
      if (cond_0 or cond_1) {
        reduce_axis[capacity_axis] = capacity_shape;
        capacity_axis++;
      }
      first = second;
      capacity_shape++;
      second += 1;
      continue;
    }
  }

  input_shape.resize(capacity_shape);
  reduce_axis.resize(capacity_axis);
}

template <typename T>
void Reduce<T>::DealDisableFuseAxes(std::vector<int32_t>& pos) {
    int32_t cnt = 1;

    bool no_reduce = reduce_axis_ori.size() == 0 ? true: false;
    bool all_reduce = reduce_axis_ori.size() == input_shape_ori.size() ? true: false;
    bool is_ori_dim_aligned = original_input_shape[compileInfo->ori_dim_index] % compileInfo->min_block_size == 0;
    bool is_nd_case = false;
    if (is_ori_dim_aligned && (no_reduce || all_reduce)) {
      is_nd_case = true;
    }

    if(!is_nd_case) {
      for (const auto& idx: compileInfo->disable_fuse_axes) {
        pos[idx] += MODULO * (cnt++);
      }
    }
}

template <typename T>
void Reduce<T>::GetReduceShapeCommonInfo() {
  total_output_count = 1;
  total_reduce_count = 1;
  output_shape.resize(input_shape.size());

  for (uint32_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      if (compileInfo->is_keep_dims) {
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
  block_size = compileInfo->min_block_size;
  is_last_axis_reduce = IsInVector(reduce_axis, input_shape.size() - 1);
}

template <typename T>
void Reduce<T>::ChooseAtomic() {
  int64_t ub_info_size = compileInfo->ub_info[reduceTilingInfo.idx];
  int64_t ub_info_rf_size = compileInfo->ub_info_rf[reduceTilingInfo.idx];
  int64_t ub_info_pad_size = compileInfo->ub_info_pad.size() == 0 ? 0 : compileInfo->ub_info_pad[reduceTilingInfo.idx];
  int64_t ub_info_transpose_size = compileInfo->ub_info_transpose.size() == 0 ?
                                                              0 : compileInfo->ub_info_transpose[reduceTilingInfo.idx];
  int64_t mib_ub_info_size = ub_info_size < ub_info_rf_size ? ub_info_size : ub_info_rf_size;
  mib_ub_info_size = ub_info_pad_size != 0 && ub_info_pad_size < mib_ub_info_size ?
                                                              ub_info_pad_size : mib_ub_info_size;
  mib_ub_info_size = ub_info_transpose_size != 0 && ub_info_transpose_size < mib_ub_info_size ?
                                                              ub_info_transpose_size : mib_ub_info_size;

  // Layer 0 Check if atomic is enabled
  bool atomic_available = compileInfo->atomic;
  // Layer 1 Check if output is large enough (> SMALL_SHAPE_THRESHOLD)
  //         Check normal atomic rules
  reduceTilingInfo.atomic = total_output_count <= mib_ub_info_size &&
                       total_output_count * total_reduce_count > SMALL_SHAPE_THRESHOLD &&
                       total_output_count < static_cast<int64_t>(compileInfo->core_num) * block_size / BASE_2 &&
                       total_reduce_count > static_cast<int64_t>(compileInfo->core_num) / BASE_2;
  // Layer 2 Check if it is nlast_reduce
  //         Check if it is in a0, r, a1 pattern and a0 is 0
  bool is_outermost_nlast_reduce = std::find(reduce_axis.begin(), reduce_axis.end(),
                                             (int32_t)1) != reduce_axis.end() &&
                                             input_shape[0] == (int32_t)1 &&
                                             std::find(reduce_axis.begin(), reduce_axis.end(),
                                                 static_cast<int32_t>(output_shape.size() - 1)) == reduce_axis.end();
  // Check if output_shape is smaller than Single Tensor Size Limitation so that r, a ub_split_a schedule won't be used
  bool output_shape_limitation = total_output_count <= mib_ub_info_size;
  // Check if outermost reduce axis is larger than or equal to core_num
  bool input_shape_limitation = input_shape[1] >= compileInfo->core_num &&
                                mib_ub_info_size / compileInfo->coef > SMALL_SHAPE_THRESHOLD * BASE_4;
  // AND expression for all checks
  bool shape_limitation = output_shape_limitation && input_shape_limitation;
  // check extracted here because of 120 characters per line static check rule
  reduceTilingInfo.atomic = reduceTilingInfo.atomic || (shape_limitation && is_outermost_nlast_reduce);
  // Final
  reduceTilingInfo.atomic = reduceTilingInfo.atomic && atomic_available;
}

template <typename T>
bool Reduce<T>::ChooseGroupAxis() {
  // dont support group reduce
  if (!compileInfo->group_reduce) {
    return false;
  }
  // ARA
  if (is_last_axis_reduce || reduce_axis.size() != 1) {
    return false;
  }
  // can not enable atomic
  if (compileInfo->atomic) {
    return false;
  }
  // don't exist 0 shape
  if (exit_zero_axis) {
    return false;
  }

  int64_t reduce_product = 1;
  int64_t common_product_except_last_a = 1;
  for (std::size_t i = 0; i < input_shape.size(); i++) {
    if (IsInVector(reduce_axis, static_cast<int32_t>(i))) {
      reduce_product *= input_shape[i];
    } else if (i != input_shape.size() - 1) {
      common_product_except_last_a *= input_shape[i];
    }
  }

  int64_t last_dim_a = is_last_axis_reduce ? 1 : input_shape.back();
  // common product(except last a) is small enough
  if (common_product_except_last_a >= compileInfo->core_num / BASE_2) {
    return false;
  }
  // reduce product is large enough
  if (reduce_product < compileInfo->core_num * REDUCE_PRODUCT_COEFFICIENT) {
    return false;
  }
  // second step don't split ub
  return compileInfo->core_num * last_dim_a <= compileInfo->ub_info[reduceTilingInfo.idx];
}

template <typename T>
bool Reduce<T>::IsReduceTransposeCase() {
  // no pad_max_entire_size given , avoid old ut error
  bool pad_max_entire_size_not_exist = compileInfo->pad_max_entire_size == -1;

  // not support pad in tilingcase
  bool ub_info_transpose_not_exist = compileInfo->ub_info_transpose.size() == 0 ||
                                     compileInfo->ub_info_transpose[reduceTilingInfo.idx] == 0;

  // only support shape size of 2
  bool input_shape_size_not_support = input_shape.size() != SHAPE_LENGTH_TWO;

  // should be last reduce
  bool not_last_reduce = !IsInVector(reduce_axis, 1);

  // should no one axis exist
  bool exist_one_axis = input_shape[0] == 1 || input_shape[1] == 1;

  // block split should be on A axis
  bool is_atomic = reduceTilingInfo.atomic;

  // should be not aligned
  bool last_axis_aligned = input_shape.back() % block_size == 0;
  if (pad_max_entire_size_not_exist || ub_info_transpose_not_exist ||  input_shape_size_not_support ||
      exist_one_axis || not_last_reduce || is_atomic || last_axis_aligned) {
      return false;
  }

  // last dim is not small enough
  int32_t pad_max_entire_size = compileInfo->pad_max_entire_size;
  int32_t last_dim = input_shape[input_shape.size() - 1];
  int32_t last_dim_align = (last_dim + block_size - 1) / block_size * block_size;
  if (compileInfo->ub_info_pad[reduceTilingInfo.idx] / pad_max_entire_size < last_dim_align) {
    return false;
  }

  // as test, more than 64 per core for last 2 axis, will increase the efficiency. both fp16 and fp32
  if (input_shape[0] / compileInfo->core_num < TRANSPOSE_THRESHOLD_VALUE) {
   return false;
  }

  return true;
}

template <typename T>
bool Reduce<T>::IsReducePadCase() const {
  // no pad_max_entire_size given , avoid old ut error
  if (-1 == compileInfo->pad_max_entire_size) {
    return false;
  }

  // not support pad in tilingcase
  if (compileInfo->ub_info_pad.size() == 0 ||
      compileInfo->ub_info_pad[reduceTilingInfo.idx] == 0) {
    return false;
  }

  // support only one reduce axis
  if (reduce_axis.size() > 1) {
    return false;
  }

  // input shape size limit in 2 and 3
  int32_t input_shape_size = input_shape.size();
  if (input_shape_size < SHAPE_LENGTH_TWO || input_shape_size > SHAPE_LENGTH_THREE) {
    return false;
  }

  // last dim is not align
  if (input_shape.back() % block_size == 0) {
    return false;
  }

  // last dim is not small enough
  int32_t pad_max_entire_size = compileInfo->pad_max_entire_size;
  int32_t last_dim = input_shape[input_shape.size() - 1];
  int32_t last_dim_align = (last_dim + block_size - 1) / block_size * block_size;
  if (compileInfo->ub_info_pad[reduceTilingInfo.idx] / pad_max_entire_size < last_dim_align) {
    return false;
  }

  // number of not 1 axis should be greater than or equal to 2
  std::size_t count_one = 0;
  for (const auto& single_dim : input_shape) {
    if (single_dim == 1) {
      count_one++;
    }
  }
  if (input_shape.size() - count_one < MIN_NOT_ONE_AXIS_NUM) {
    return false;
  }

  return IsEnableReducePad();
}

template <typename T>
bool Reduce<T>::IsEnableReducePad() const {
  // group reduce can enable align pad
  if (reduceTilingInfo.group_reduce) {
    return true;
  }
  
  int32_t input_shape_size = input_shape.size();
  int32_t core_num = compileInfo->core_num;
  int32_t pad_max_entire_size = compileInfo->pad_max_entire_size;
  if ((input_shape_size == SHAPE_LENGTH_TWO &&
                !reduceTilingInfo.atomic && input_shape[0] / core_num >= pad_max_entire_size)
      || (input_shape_size == SHAPE_LENGTH_THREE && ((reduceTilingInfo.atomic && input_shape[1] / core_num >=
      pad_max_entire_size)||
      (!reduceTilingInfo.atomic && input_shape[0] != 1 &&
      ((input_shape[0] >= core_num && input_shape[0] / core_num * input_shape[1] >= pad_max_entire_size)
      || (input_shape[0] < core_num && input_shape[INDEX_OF_LAST_DIM_OF_ARA_CASE] < block_size
      && input_shape[1] >= pad_max_entire_size)))))) {
    return true;
  }

  return false;
}

template <typename T>
void Reduce<T>::ChooseUBInfo() {
  // UB_SPACE of ub_info is more than ub_info_rf, Atomic selected the former.
  // nodes after reduce(include reduce) have same space(ubSizeA)
  // nodes before reduce have same space(ubSizeB)
  // ubSizeB = ubSizeA * coef (max dtype)
  reduceTilingInfo.max_ub_count =
  ubSizeB = compileInfo->ub_info[reduceTilingInfo.idx];

  // According adaptation of SCH, choose the best UBInfo.
  // Rfactor only attached in Last Reduce.
  if (compileInfo->support_transpose) {
    is_reduce_transpose_case = IsReduceTransposeCase();
  }

  // if match transpose case, won't match pad case.
  if (!is_reduce_transpose_case) {
    is_reduce_pad_case = IsReducePadCase();
  }

  if (is_reduce_transpose_case) {
    ubSizeB = compileInfo->ub_info_transpose[reduceTilingInfo.idx];
    reduceTilingInfo.sch_type = REDUCE_TRANSPOSE_SCH_TYPE;
  } else if (is_reduce_pad_case) {
    ubSizeB = compileInfo->ub_info_pad[reduceTilingInfo.idx];
    reduceTilingInfo.sch_type = REDUCE_PAD_SCH_TYPE;
  } else if (is_last_axis_reduce) {
    int64_t last_dim = input_shape[input_shape.size()-1];
    int64_t real_reduce_count = total_reduce_count / last_dim;
    last_dim = (last_dim + block_size - 1) / block_size * block_size;
    real_reduce_count *= last_dim;
    bool ub_split_in_r = real_reduce_count > ubSizeB;
    if (ub_split_in_r) {
      ubSizeB = compileInfo->ub_info_rf[reduceTilingInfo.idx];
    }
  }
  ubSizeA = ubSizeB / compileInfo->coef;
}

template <typename T>
bool Reduce<T>::GetUbTilingInfo() {
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

      // if is AR pattern and both block and ub split on A, it will be transpose case
      // if block factor > ub factor, should do align to ub factor even to zero to
      // avoid some pass align error
      if (is_reduce_transpose_case && reduceTilingInfo.block_tiling_factor
                                            > reduceTilingInfo.ub_tiling_factor) {
        reduceTilingInfo.ub_tiling_factor = reduceTilingInfo.ub_tiling_factor / block_size * block_size;
      }

      return true;
    }
  }

  return false;
}

template <typename T>
void Reduce<T>::ProcessReorderAxis(int32_t fused_type) {
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

template <typename T>
bool Reduce<T>::GetAtomicBlockDim() {
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

template <typename T>
bool Reduce<T>::GetAtomicBlockTilingInfo() {
  // rewrite block_tiling_axis, block_tiling_factor.
  bool is_find_block_tiling = false;
  int64_t left_mul = 1;
  int32_t core_num = compileInfo->core_num;
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

template <typename T>
void Reduce<T>::GetNotMulCoreBlockTiling() {
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

template <typename T>
bool Reduce<T>::GetBlockTilingInfoLessThanCoreNum(int32_t left_block_dim, uint32_t i, int64_t right_total_num) {
  if (left_block_dim > 1) {
    int64_t cur_block_factor = (block_size + right_total_num - 1) / right_total_num;
    if (GetBlockTilingInfoX(cur_block_factor, i, right_total_num)) {
      return true;
    }
  } else {
    int64_t cur_block_factor = (block_size + right_total_num - 1) / right_total_num;
    reduceTilingInfo.block_tiling_axis = i;
    reduceTilingInfo.block_tiling_factor = cur_block_factor;
    return true;
  }
  return false;
}

template <typename T>
bool Reduce<T>::GetBlockTilingInfo() {
  int32_t left_block_dim = 1;
  int64_t max_ub_count = ubSizeA;
  int32_t core_num = compileInfo->core_num;

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
      if(GetBlockTilingInfoLessThanCoreNum(left_block_dim, i, right_total_num)) {
        return true;
      }
    } else if (right_total_num <= block_size || left_block_dim * output_shape[i] >= core_num) {
      if (GetBlockTilingInfoY(left_block_dim, i, max_block_tilling_factor, right_total_num)) {
        return true;
      }
    }

    left_block_dim = (int32_t)left_block_dim * output_shape[i];
  }
  return false;
}

template <typename T>
bool Reduce<T>::GetBlockTilingInfoX(int64_t cur_block_factor, int64_t i, int64_t right_total_num) {
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

template <typename T>
bool Reduce<T>::GetBlockTilingInfoInner(int64_t i, int64_t cur_block_dim,
                                        int64_t cur_block_factor, int64_t right_total_num) {
  int64_t tail_block_factor =
      output_shape[i] % cur_block_factor == 0 ? cur_block_factor : output_shape[i] % cur_block_factor;
  int64_t tail_block_tilling_inner_ddr_count = tail_block_factor * right_total_num;
  if (tail_block_tilling_inner_ddr_count < block_size) {
    for (cur_block_dim = cur_block_dim - 1; cur_block_dim >= 1; cur_block_dim--) {
      if (cur_block_dim == 0) {
        return false;
      }
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
  return false;
}

template <typename T>
bool Reduce<T>::GetBlockTilingInfoY(int32_t left_block_dim, int64_t i,
                                    int64_t max_block_tilling_factor, int64_t right_total_num) {
  int32_t core_num = compileInfo->core_num;
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
          if (GetBlockTilingInfoInner(i, cur_block_dim, cur_block_factor, right_total_num)) {
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

template <typename T>
bool Reduce<T>::ProcessAtomicTiling() {
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

template <typename T>
bool Reduce<T>::ProcessGroupTiling() {
  bool ret = GetGroupBlockTilingInfo();
  ret = ret && GetGroupUbTilingInfo();

  return ret;
}

template <typename T>
bool Reduce<T>::GetGroupBlockTilingInfo() {
  int64_t left_product = 1;
  for (int32_t i = 0; i < static_cast<int32_t>(input_shape.size()); i++) {
    // block split R
    if (!IsInVector(reduce_axis, i)) {
      left_product *= input_shape[i];
      continue;
    }
    if (left_product < compileInfo->core_num && left_product * input_shape[i] >= compileInfo->core_num) {
      reduceTilingInfo.block_tiling_axis = i;
      int64_t need_block_outer = compileInfo->core_num / left_product;
      reduceTilingInfo.block_tiling_factor = (input_shape[i] + need_block_outer - 1) / need_block_outer;
      reduceTilingInfo.block_dim = (input_shape[i] + reduceTilingInfo.block_tiling_factor - 1) /
        reduceTilingInfo.block_tiling_factor * left_product;
      return true;
    } else {
      left_product *= input_shape[i];
    }
  }

  return false;
}

template <typename T>
bool Reduce<T>::GetGroupUbTilingInfo() {
  int64_t right_product = 1;
  for (int32_t i = static_cast<int32_t>(input_shape.size()) - 1; i >= 0; i--) {
    if (right_product < ubSizeB && right_product * input_shape[i] >= ubSizeB) {
      // find ub split axis
      reduceTilingInfo.ub_tiling_axis = i;
      reduceTilingInfo.ub_tiling_factor = ubSizeB / right_product;
      return true;
    } else if (i == reduceTilingInfo.block_tiling_axis) {
      // first optional axis is block_tiling_axis
      reduceTilingInfo.ub_tiling_axis = i;
      reduceTilingInfo.ub_tiling_factor = reduceTilingInfo.block_tiling_factor;
      return true;
    } else {
      if (i == static_cast<int32_t>(input_shape.size()) - 1) {
        right_product = (input_shape[i] + block_size - 1) / block_size * block_size * right_product;
      } else {
        right_product = input_shape[i] * right_product;
      }
    }
  }

  return false;
}

template <typename T>
bool Reduce<T>::ProcessNormalTiling() {
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

  if (!compileInfo->is_keep_dims) {
    reduceTilingInfo.block_tiling_axis = GetRealBlockTilingAxis(output_shape, reduceTilingInfo.block_tiling_axis);
  }

  return true;
}

template <typename T>
bool Reduce<T>::FineTuning() {
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

template <typename T>
bool Reduce<T>::IsZero() {
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

template <typename T>
bool Reduce<T>::DoZeroBranch() {
  if (exit_non_reduce_zero_axis) {
    // EmptySchedule
    zero_tiling_key = MAX_INTEGER;
    reduceTilingInfo.ub_tiling_factor = EMPTY_SCHEDULE_UB_TILING_FACTOR_128;
  } else {
    zero_tiling_key = BASE_10;
    reduceTilingInfo.ub_tiling_factor = compileInfo->zero_ub_factor;
    V_OP_TILING_CHECK(reduceTilingInfo.ub_tiling_factor > 0,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                      "zero_ub_factor is %ld, which is init_value",
                      reduceTilingInfo.ub_tiling_factor),
                      return false);
  }
  return true;
}

template <typename T>
bool Reduce<T>::DoConstRunTimeBranch() {
  // runtime in DoTiling while input is const
  EliminateOne();
  pattern = CalcConstPattern(reduce_axis_ori);
  std::string pattern_str = std::to_string(pattern);
  reduceTilingInfo.const_block_dims = compileInfo->block_dim_map.at(pattern_str);
  reduceTilingInfo.const_atomic_flag = compileInfo->atomic_flags_map.at(pattern_str);
  return true;
}

template <typename T>
bool Reduce<T>::WriteWorkspace() {
  if (compileInfo->workspace_size == 0) {
    return true;
  }

  std::size_t workspace_len = 0;
  std::array<int64_t, REDUCE_MAX_WORKSPACE_NUMS> workspace{};
  if (reduceTilingInfo.group_reduce) {
    int64_t right_product = 1;
    for (int32_t i = static_cast<int32_t>(input_shape.size()) - 1; i >= 0; i--) {
      if (i == reduceTilingInfo.block_tiling_axis) {
        break;
      }
      if (i == static_cast<int32_t>(input_shape.size()) - 1) {
        right_product = (input_shape[i] + block_size - 1) / block_size * block_size * right_product;
      } else {
        right_product = input_shape[i] * right_product;
      }
    }
    int32_t actual_reduce_type_bytes = compileInfo->reduce_dtype_byte == -1 ?
                                       ALIGN_BYTES / block_size :
                                       compileInfo->reduce_dtype_byte;
    // workspace tensor
    workspace[workspace_len] = right_product * actual_reduce_type_bytes * compileInfo->core_num;
    workspace_len++;
    if (compileInfo->workspace_size > 1) {
      // sync tensor
      workspace[workspace_len] = SINGLE_SYNC_CORE_BYTES * compileInfo->core_num;
      workspace_len++;
    }
  } else {
    // fake workspace
    for (int32_t i = 0; i < compileInfo->workspace_size; ++i) {
      workspace[i] = FAKE_WORKSPACE_SIZE;
    }
    workspace_len = compileInfo->workspace_size;
  }
  context->AddWorkspace(workspace.begin(), workspace_len);
  return true;
}

template <typename T>
bool Reduce<T>::WriteTilingData() {
  context->SetNeedAtomic(reduceTilingInfo.atomic || reduceTilingInfo.group_reduce);

  if (reduceTilingInfo.atomic) {
    context->SetNeedAtomic(true);
  } else {
    context->SetNeedAtomic(false);
  }

  if (exit_zero_axis) {
    context->Append((int32_t)fusion_dim_value);
    context->Append((int32_t)reduceTilingInfo.ub_tiling_factor);
    context->SetBlockDim(1);
    uint32_t zero_tiling_key_uint = static_cast<uint32_t>(zero_tiling_key);
    context->SetTilingKey(zero_tiling_key_uint);

    return WriteWorkspace() && context->WriteVarAttrs(zero_tiling_key_uint);
  }

  if (compileInfo->is_const_post) {
    // runtime
    context->SetBlockDim(reduceTilingInfo.const_block_dims);
    context->SetNeedAtomic(reduceTilingInfo.const_atomic_flag);
    uint32_t pattern_uint = static_cast<uint32_t>(pattern);
    context->SetTilingKey(pattern_uint);

    return context->WriteVarAttrs(pattern_uint);
  }

  if (compileInfo->is_const) {
    // compile
    return WriteConstTilingData();
  }

  return WriteDynamicTilingData();
}

template <typename T>
bool Reduce<T>::WriteConstTilingData() {
    // compile
    context->Append((int32_t)reduceTilingInfo.block_tiling_axis);
    context->Append((int32_t)reduceTilingInfo.block_tiling_factor);
    context->Append((int32_t)reduceTilingInfo.ub_tiling_axis);
    context->Append((int32_t)reduceTilingInfo.ub_tiling_factor);
    context->Append(static_cast<int32_t>(ubSizeB));
    context->Append(static_cast<int32_t>(ubSizeA));
    context->Append(reduceTilingInfo.sch_type);
    context->Append(static_cast<int32_t>(reduceTilingInfo.group_reduce));
    context->SetBlockDim(static_cast<uint32_t>(reduceTilingInfo.block_dim));
    return true;
}

template <typename T>
bool Reduce<T>::GetVarValue(uint64_t tiling_key) {
  var_value_len = 0;
  try {
    int32_t ub_tiling_factor_encode = 40000;
    int32_t block_tiling_factor_encode = 30000;
    int32_t reduce_dim_encode = 20000;
    int32_t reduce_ori_dim_encode = 10000;

    const auto& var_pattern = compileInfo->reduce_vars.at(tiling_key);
    for (const auto& var: var_pattern) {
      if (var >=ub_tiling_factor_encode) {
        var_value[var_value_len] = reduceTilingInfo.ub_tiling_factor;
      } else if (var >=block_tiling_factor_encode) {
        var_value[var_value_len] = reduceTilingInfo.block_tiling_factor;
      } else if (var >=reduce_dim_encode) {
        var_value[var_value_len] = input_shape[var % reduce_dim_encode];
      } else {
        var_value[var_value_len] = original_input_shape[var % reduce_ori_dim_encode];
      }
      var_value_len++;
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type,"get var value error. Error message: %s", e.what());
    return false;
  }
  return true;
}


template <typename T>
bool Reduce<T>::WriteDynamicTilingData() {
  // tiling_key
  context->SetBlockDim(static_cast<uint32_t>(reduceTilingInfo.block_dim));
  int32_t tiling_key = CalcTilingKey();
  uint32_t tiling_key_uint = static_cast<uint32_t>(tiling_key);
  context->SetTilingKey(tiling_key_uint);

  if(!GetVarValue(tiling_key_uint)) {
    return false;
  }

  for (std::size_t i = 0; i < var_value_len; i++) {
    context->Append(static_cast<int32_t>(var_value[i]));
  }
  
  OP_LOGD(op_type.c_str(), "block/res_ub tilling axis:%d", reduceTilingInfo.block_tiling_axis);
  OP_LOGD(op_type.c_str(), "block/res_ub tilling factor:%d", reduceTilingInfo.block_tiling_factor);
  OP_LOGD(op_type.c_str(), "ub/input_ub tilling axis:%d", reduceTilingInfo.ub_tiling_axis);
  OP_LOGD(op_type.c_str(), "ub/input_ub tilling factor:%d", reduceTilingInfo.ub_tiling_factor);

  return WriteWorkspace() &&  context->WriteVarAttrs(tiling_key_uint);
}

template <typename T>
bool Reduce<T>::CheckCompileInfoForCalculate() {
  // Required info from SCH that do for calculating
  V_CHECK_EQ(compileInfo->pattern_info.size(), compileInfo->ub_info.size(),
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "pattern_info's size should be as same as ub_info"),
               return false);
  V_CHECK_EQ(compileInfo->pattern_info.size(), compileInfo->ub_info_rf.size(),
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "pattern_info's size should be as same as ub_info_rf"),
               return false);

  // CHECK VALUE
  V_OP_TILING_CHECK(!(compileInfo->coef <= 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "coef is %d that is illegal", compileInfo->coef),
                      return false);
  V_OP_TILING_CHECK(!(compileInfo->min_block_size <= 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                      "min_block_size is %d that is illegal",
                                                      compileInfo->min_block_size),
                      return false);
  V_OP_TILING_CHECK(!(compileInfo->core_num <= 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "core_num is %d that is illegal", compileInfo->core_num),
                      return false);
  return true;
}

template <typename T>
bool Reduce<T>::GetGeInfo() {
  // Get Input
  uint32_t inputs_num = context->GetInputNums();
  V_OP_TILING_CHECK(!(inputs_num == 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "inputs cannot be empty"),
                    return false);
  V_OP_TILING_CHECK(!(inputs_num <= compileInfo->idx_before_reduce),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "idx is invalid index for inputs"),
                    return false);

  if (!GetInputShapeOri()) {
    return false;
  }
  return true;
}

template <typename T>
bool Reduce<T>::SetInit() {
  // Get ReduceAxis
  bool ret = GetReduceAxisTensor();
  if (!ret) {
    return false;
  }
  // Convert reduce axis (-1 -> length+1)
  int32_t max_value = int32_t(input_shape_ori.size());
  int32_t min_value = -1 * max_value;
  for (size_t i = 0; i < reduce_axis_ori.size(); i++) {
    if (reduce_axis_ori[i] >= max_value || reduce_axis_ori[i] < min_value) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of axis is %ld which exceed max_value: %d or min_value: %d",
                                      reduce_axis_ori[i], max_value, min_value);
      return false;
    }
    if (reduce_axis_ori[i] < 0) {
      reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
    }
  }

  if(!GetOriginInputShape()) {
    return false;
  }
  return true;
}

template <typename T>
bool Reduce<T>::MatchPattern() {
  for (auto item: compileInfo->pattern_info) {
    if (item == pattern) {
      break;
    }
    reduceTilingInfo.idx += 1;
  }

  // CHECK VALUE
  if (reduceTilingInfo.idx >= compileInfo->pattern_info.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "pattern is %d that not in pattern_info", pattern);
    return false;
  }

  return true;
}

template <typename T>
bool Reduce<T>::TilingProcess() {
  if (reduceTilingInfo.atomic) {
    return ProcessAtomicTiling();
  } else if (reduceTilingInfo.group_reduce) {
    return ProcessGroupTiling();
  } else {
    return ProcessNormalTiling();
  }
}

template <typename T>
bool Reduce<T>::DoReduceTiling() {
  /* Situations of DoTiling include:
     1. input(known):
        status of compile: do others except FusedReduceAxis
        status of runtime: do WriteTilingData
     2. input(unknown):
        do all process
  */
  compileInfo = dynamic_cast<const ReduceCompileInfo *>(context->GetCompileInfo());
  V_OP_TILING_CHECK(GetGeInfo(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetGeInfo Failed"),
                    return false);
  V_OP_TILING_CHECK(SetInit(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "SetInit Failed"),
                    return false);
  if (IsZero()) {
    return DoZeroBranch();
  }

  if (compileInfo->is_const && compileInfo->is_const_post) {
    // status: runtime
    return DoConstRunTimeBranch();
  }

  if (compileInfo->is_const) {
    // status: compile
    V_OP_TILING_CHECK(compileInfo->compile_pattern.first,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get compile_pattern Failed"),
                      return false);
    pattern = compileInfo->compile_pattern.second;
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
  GetReduceShapeCommonInfo();
  ChooseAtomic();
  reduceTilingInfo.group_reduce = ChooseGroupAxis();
  ChooseUBInfo();

  V_OP_TILING_CHECK(TilingProcess(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TilingProcess Failed"),
                    return false);
  V_OP_TILING_CHECK(FineTuning(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "FineTuning Failed"),
                    return false);

  return true;
}

template <typename T>
bool Reduce<T>::GetReduceAxisTensor() {
  // Get ReduceAxis
  if (compileInfo->reduce_axes_type == REDUCE_AXES_TYPE_ALL) {
    size_t input_shape_ori_size = input_shape_ori.size();
    reduce_axis_ori.resize(input_shape_ori_size);
    for (size_t i = 0; i < input_shape_ori_size; i++) {
      reduce_axis_ori[i] = i;
    }
  } else if (compileInfo->ori_axis.first) {
    reduce_axis_ori = compileInfo->ori_axis.second;
  } else {
    // axes is tensor
    // te_fusion will convert axes_idx while do fusion in reduce_op
    uint32_t axes_idx = compileInfo->axes_idx.first ? compileInfo->axes_idx.second:MAX_INTEGER;
    V_OP_TILING_CHECK(context->GetConstInput("axes", axes_idx, reduce_axis_ori),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get reduce axes by index %d failed.", axes_idx),
                      return false);
  }

  return true;
}

template <typename T>
bool Reduce<T>::DoReduceTiling(const OpInfoImpl& op_info) {
  /* Situations of DoTiling include:
     1. input(known):
        status of compile: do others except FusedReduceAxis
        status of runtime: do WriteTilingData
     2. input(unknown):
        do all process
  */
  // Get Input
  compileInfo = dynamic_cast<const ReduceCompileInfo *>(context->GetCompileInfo());
  if (!GetInputShapeOri(op_info)) {
    return false;
  }

  if (!GetOriginInputShape()) {
    return false;
  }

  if (!GetReduceAxisOri(op_info)) {
    return false;
  }

  if (IsZero()) {
    return DoZeroBranch();
  }

  if (compileInfo->is_const && compileInfo->is_const_post) {
    // status: runtime
    return DoConstRunTimeBranch();
  }

  if (compileInfo->is_const) {
    // status: compile
    V_OP_TILING_CHECK(compileInfo->compile_pattern.first,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get compile_pattern Failed"), return false);
    pattern = compileInfo->compile_pattern.second;
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
  GetReduceShapeCommonInfo();
  ChooseAtomic();
  reduceTilingInfo.group_reduce = ChooseGroupAxis();
  ChooseUBInfo();
  V_OP_TILING_CHECK(TilingProcess(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TilingProcess Failed"), return false);
  V_OP_TILING_CHECK(FineTuning(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "FineTuning Failed"), return false);

  return true;
}

template <typename T>
bool Reduce<T>::GetInputShapeOri(const OpInfoImpl& op_info) {
  const std::vector<std::vector<int64_t>>* op_input_shape = op_info.GetInputShape();
  if (op_input_shape != nullptr && !op_input_shape->empty()) {
    input_shape_ori = op_input_shape->at(0);
  } else {
    uint32_t inputs_num = context->GetInputNums();
    V_OP_TILING_CHECK(!(inputs_num == 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "inputs cannot be empty"),
                      return false);
    V_OP_TILING_CHECK(!(inputs_num <= compileInfo->idx_before_reduce),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "idx is invalid index for inputs"),
                      return false);
    if (!GetInputShapeOri()) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool Reduce<T>::GetReduceAxisOri(const OpInfoImpl& op_info) {
  const std::vector<int64_t>* axes = op_info.GetAxes();
  if (axes != nullptr && !axes->empty()) {
    int32_t reduce_axis_length = axes->size();
    reduce_axis_ori.resize(reduce_axis_length);
    reduce_axis_ori = *(axes);
  } else {
    bool ret = GetReduceAxisTensor();
    if (!ret) {
      return false;
    }
  }

  // Convert reduce axis (-1 -> length+1)
  int32_t max_value = int32_t(input_shape_ori.size());
  int32_t min_value = -1 * max_value;
  for (size_t i = 0; i < reduce_axis_ori.size(); i++) {
    if (reduce_axis_ori[i] >= max_value || reduce_axis_ori[i] < min_value) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of axis is %ld which exceed max_value: %d or min_value: %d",
                                      reduce_axis_ori[i], max_value, min_value);
      return false;
    }
    if (reduce_axis_ori[i] < 0) {
      reduce_axis_ori[i] = input_shape_ori.size() + reduce_axis_ori[i];
    }
  }
  return true;
}

template <typename T>
bool Reduce<T>::GetInputShapeOri() {
  OpShape input_shape = context->GetInputShape(compileInfo->idx_before_reduce);
  V_OP_TILING_CHECK(!input_shape.Empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input shape error"),
                    return false);
  auto dim_len = static_cast<int64_t>(input_shape.GetDimNum());
  input_shape_ori.resize(dim_len);
  for (int i = 0; i < dim_len; i++) {
    input_shape_ori[i] = input_shape.GetDim(i);
  }
  return true;
}

template <typename T>
bool Reduce<T>::GetOriginInputShape() {
  if (compileInfo->disable_fuse_axes.empty()) {
    OP_LOGD(op_type.c_str(), "No need get original input shape when disable_fuse_axes is empty,so return.");
    return true;
  }

  OpShape origin_shape = context->GetOriginInputShape(compileInfo->idx_before_reduce);
  V_OP_TILING_CHECK(!origin_shape.Empty(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get origin input shape error."),
                    return false);
  auto dim_len = static_cast<int64_t>(origin_shape.GetDimNum());
  original_input_shape.resize(dim_len);
  for (int i = 0; i < dim_len; i++) {
    original_input_shape[i] = origin_shape.GetDim(i);
  }
  return true;
}


template <typename T>
bool Reduce<T>::DoTiling() {
  bool ret = DoReduceTiling();
  ret = ret && WriteTilingData();
  return ret;
}

template <typename T>
bool Reduce<T>::DoTiling(const OpInfoImpl& op_info) {
  bool ret = DoReduceTiling(op_info);
  ret = ret && WriteTilingData();
  return ret;
}
} // namespace v3

bool ReduceTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "tiling running");
  AutoTilingOp auto_tiling_op(op_type.c_str(), &op_paras, &compileInfo, &run_info);
  v3::Reduce<AutoTilingOp> reduce(&auto_tiling_op, nullptr);
  return reduce.DoTiling();
}

bool ReduceTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                   const OpInfo& op_info) const {
  OP_LOGD(op_type.c_str(), "user-defined shape tiling running");
  AutoTilingOp auto_tiling_op(op_type.c_str(), &op_paras, &compileInfo, &run_info);
  const OpInfoImpl* op_info_impl = OpInfoImplGetter::GetOpInfoImpl(&op_info).get();
  v3::Reduce<AutoTilingOp> reduce(&auto_tiling_op, op_info_impl);
  return reduce.DoTiling(*op_info_impl);
}

bool CreateReduceDslTiling(gert::TilingContext* context, const OpInfoImpl* op_info) {
  AutoTilingContext auto_tiling_context(context);
  if (op_info) {
    OP_LOGD("reduce", "user-defined shape tiling running rt2");
    auto_tiling_context.SetCompileInfo(op_info->GetCompileInfo());
    v3::Reduce<AutoTilingContext> reduce(&auto_tiling_context, op_info);
    return reduce.DoTiling(*op_info);
  } else {
    OP_LOGD("reduce", "tiling running rt2");
    v3::Reduce<AutoTilingContext> reduce(&auto_tiling_context, op_info);
    return reduce.DoTiling();
  }
}

AutoTilingCompileInfo* CreateReduceDslParser(const char* op_type, const nlohmann::json& json_compile_info) {
  v3::ReduceCompileInfo* reduce_compile_info = new v3::ReduceCompileInfo(op_type, json_compile_info);
  if (!reduce_compile_info->parsed_success) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Reduce parse compile info failed");
    return nullptr;
  }
  return reduce_compile_info;
}

std::shared_ptr<AutoTilingHandler> CreateReduceTilingHandler(const std::string& op_type,
                                                             const std::string& pattern,
                                                             const nlohmann::json& parsed_compile_info) {
  auto reduceCompileInfoV3_ptr = std::make_shared<ReduceTilingHandler>(op_type, pattern, parsed_compile_info);

  return reduceCompileInfoV3_ptr->ParsedSuccess() ?
                        reduceCompileInfoV3_ptr : std::shared_ptr<AutoTilingHandler>(nullptr);
}

REGISTER_AUTO_TILING(SchPattern::COMMONREDUCE, CreateReduceDslTiling, CreateReduceDslParser);
}  // namespace optiling