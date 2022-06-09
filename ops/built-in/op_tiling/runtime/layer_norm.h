/* Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file layer_norm.h
 * \brief
 */
#ifndef LAYER_NORM_H_
#define LAYER_NORM_H_
#include <string>
#include <vector>

#include "graph/utils/op_desc_utils.h"
#include "auto_tiling_rt2.h"

namespace optiling {
struct LayerNormOpInfo {
  bool is_support_vexp_pattern;
  std::shared_ptr<AutoTilingCompileInfo> dsl_compile_info;
  std::vector<int32_t> ori_reduce_axis;
  string input_format;
  int32_t core_num;
  int32_t begin_norm_axis;
  int32_t begin_params_axis;
  bool is_tik_support;
  string tik_mode;
  int32_t ub_max_byte;
  bool atomic_clean_diff_shape;
  bool is_support_vexp;
  string reduce_mean_cof_dtype;
  std::vector<int32_t> common_info;
  std::vector<int32_t> pattern_info;
  std::vector<int32_t> ub_info;
  std::vector<int32_t> reduce_axis;
  int32_t max_ub_size_normal_fp16;
  int32_t max_ub_size_normal_fp32;
  string mode;
  bool is_unknown_mode;
  ge::DataType reduce_mean_cof_ge_dtype;
};
}  // namespace optiling
#endif  // LAYER_NORM_H_
