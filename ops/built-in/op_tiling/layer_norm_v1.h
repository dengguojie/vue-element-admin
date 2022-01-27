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
 * \file layer_norm_v1.h
 * \brief
 */
#ifndef LAYER_NORM_V1_H_
#define LAYER_NORM_V1_H_
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "external/graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "vector_tiling.h"

namespace optiling {
const int32_t REDUCE_MEAN_COF_FP32 = 1;
const int32_t REDUCE_MEAN_COF_FP16 = 2;
const int32_t NUM_TW = 2;
const int32_t NUM_THR = 3;
const int32_t CI_CORE_NUM_INDEX = 0;
const int32_t CI_CORE_KENEP_DIM_INDEX = 1;
const int32_t CI_MIN_BLOCK_SIZE = 2;
const int32_t CI_ATOMIC = 3;
const string TSCONST = "const";
const string TSDYNAMIC = "dynamic";
const int32_t LAST_DIM_RANGE_ONE = 1;
const int32_t LAST_DIM_RANGE_NUM1 = 64;
const int32_t LAST_DIM_RANGE_NUM2 = 2000;
const int32_t LAST_DIM_RANGE_ONE_KEY = 5;
const int32_t LAST_DIM_RANGE_NUM1_KEY = 0;
const int32_t LAST_DIM_RANGE_NUM2_KEY = 1;
const int32_t LAST_DIM_RANGE_OTHER_KEY = 2;
const int32_t BLOCK_NUM_16 = 16;
const int32_t BLOCK_NUM_32 = 8;
const int32_t TYPE_SIZE_16 = 2;
const int32_t TYPE_SIZE_32 = 4;
const int32_t NUM_BSA = 100;
const int32_t NUM_USA = 10;
const int32_t BLOCK_BYTE_SIZE = 32;
const int32_t TILING_FACTOR_8 = 8;
const int32_t TILING_DIVIDE_2 = 2;

struct layerNormOpInfo {
  bool is_support_vexp_pattern;
  std::shared_ptr<AutoTilingHandler> tiling_handler;
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
};

bool LayerNormTilingV1(const string &op_type, const ge::Operator &op_paras, const layerNormOpInfo &op_info,
                       utils::OpRunInfo &run_info);

}  // namespace optiling
#endif  // LAYER_NORM_V1_H_
